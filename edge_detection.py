# Standard library imports
import cv2
import numpy as np
import tkinter as tk
from functools import lru_cache
from typing import Optional, Tuple, Dict, Callable, Any, List
from tkinter import messagebox

# Local imports
from processing.contour_processing import ContourProcessor

# Local module imports
from utils.logger import log_info, log_error, log_warning, log_debug
from utils.error_handler import error_handler, validate_image_data
from utils.performance import performance_timer, memory_cleanup

class EdgeDetector: # creating a class Edgedetection
    """
    Enhanced edge detection processor with multiple algorithms and visualization options.
    
    Supports 6 edge detection algorithms: Canny, Sobel, Laplacian, Prewitt, Scharr, and Roberts.
    Includes post-processing features like edge inversion, thickness adjustment, and color overlay.
    
    Special optimizations for when color selection is NOT used but contour detection
    with edge detection source and area filtering is enabled - this combination can
    cause performance issues, so enhanced safeguards and limits are applied.
    """
    
    def __init__(self, app): # this is the constructor which will execute at the beginning when we call the class
        """Initialize the edge detector with algorithm registry."""
        self.app = app
        self._edge_algorithms: Dict[str, Callable] = {
            "Canny": self._apply_canny,
            "Sobel": self._apply_sobel,
            "Laplacian": self._apply_laplacian,
            "Prewitt": self._apply_prewitt,
            "Scharr": self._apply_scharr,

            "Roberts": self._apply_roberts
        }
        self.contour_processor = ContourProcessor(app) # calls the CounterProcessor class from the contour_processing.py file
        log_debug("EdgeDetector initialized", algorithms=list(self._edge_algorithms.keys())) # calls the function log_debug from the error_handler file in the utils folder
    
    @error_handler(error_message="Failed to apply edge detection. Please check your image and settings.")
    @performance_timer("edge_detection")
    def apply_edge_detection(self, *args, fast_mode: bool = False) -> None:
        """
        Apply edge detection algorithm with current settings.
        
        This method orchestrates the complete edge detection pipeline:
        1. Source image selection (original/enhanced/masked)
        2. Preprocessing (grayscale conversion, blur)
        3. Edge detection algorithm application
        4. Post-processing (inversion, thickness)
        5. Visualization (color overlay, contours)
        6. Display update
        
        Args:
            fast_mode: If True, skip expensive operations like contour detection and UI updates
                      for better performance during bulk operations like reset settings
        """
        import time
        start_time = time.time() # this will store the exact time when this function is called (when the main file is executed)
        
        if self.app.original_image is None:
            log_warning("No image loaded for edge detection") # if there is no image read from the main.py file, it returns an error
            return
        
        # Skip contour detection in fast mode
        contour_enabled = False if fast_mode else self._is_contour_enabled()
        if contour_enabled:
            self._show_contour_loader_safe("Processing contours...")
            
        edge_method = self.app.edge_method.get()
        log_debug(f"Applying edge detection with method: {edge_method}" + (" (fast mode)" if fast_mode else ""))
        
        # Skip complex checks in fast mode
        if not fast_mode:
            # Check for the problematic combination early and warn user
            if contour_enabled:
                contour_source = self.app.contour_source.get()
                area_filter_enabled = self.app.area_filter_enabled.get()
                object_selection_active = (hasattr(self.app, 'masked_image') and 
                                         self.app.masked_image is not None)
                
                if (not object_selection_active and 
                    contour_source == "edges" and 
                    area_filter_enabled):
                    log_info("Processing with color selection disabled + edge detection + area filter - performance optimizations enabled")
                    # Set status to inform user
                    self.app.status_var.set("Processing edge detection + contour detection (optimized for current settings)...")
        
        # Check for timeout (skip in fast mode)
        if not fast_mode and time.time() - start_time > 10.0:  # 10 second timeout
            log_error("Edge detection timeout - operation taking too long")
            return
        
        # Get source image
        source_img = self._get_source_image()
        if source_img is None:
            return
        
        # Validate image data
        validate_image_data(source_img)
        
        # Check for timeout (skip in fast mode)
        if not fast_mode and time.time() - start_time > 10.0:
            log_error("Edge detection timeout after image validation")
            return
        
        # Convert to grayscale and apply blur
        gray = self._preprocess_image(source_img)
            
        # Apply the selected edge detection algorithm
        edges = self._apply_edge_algorithm(edge_method, gray)
            
        # Post-process the edges (invert, thickness)
        edges = self._post_process_edges(edges)
        
        # Check for timeout before visualization (skip in fast mode)
        if not fast_mode and time.time() - start_time > 10.0:
            log_error("Edge detection timeout before visualization")
            return
        
        # Apply visualization (color overlay, contours, etc.)
        if fast_mode:
            # In fast mode, skip contour detection and just apply basic visualization
            if self.app.color_overlay.get():
                self._apply_color_overlay(edges, source_img)
            else:
                self.app.processed_image = edges
        else:
            self._apply_visualization(edges, source_img)
            
        # Display the result
        self._update_display()
        
        total_time = time.time() - start_time
        log_info(f"Edge detection completed successfully using {edge_method} in {total_time:.3f}s" + (" (fast mode)" if fast_mode else ""))
        
        # Skip UI updates in fast mode
        if not fast_mode:
            # Update UI safely in main thread to prevent crashes
            if hasattr(self.app, 'root') and self.app.root:
                try:
                    # Use after() to ensure UI updates happen in main thread
                    self.app.root.after(0, self._safe_update_ui_after_completion)
                except Exception as e:
                    log_error("Error scheduling UI update", exception=e)
        
        # Hide loader when processing is complete
        if contour_enabled:
            self._hide_contour_loader_safe()
        
        # Clean up memory if needed (skip in fast mode for speed)
        if not fast_mode:
            memory_cleanup()
            
            # Force garbage collection for large images to prevent memory issues
            if hasattr(self.app, 'original_image') and self.app.original_image is not None:
                image_size = self.app.original_image.shape[0] * self.app.original_image.shape[1]
                if image_size > 5000000:  # 5MP+
                    import gc
                    gc.collect()
                    log_debug("Forced garbage collection for large image")
            
            # Update toolbar button states after edge detection
            if hasattr(self.app, 'toolbar'):
                self.app.toolbar.force_state_update()
    
    def _is_contour_enabled(self) -> bool:
        """Check if contour detection is enabled via the checkbox."""
        if hasattr(self.app, 'control_panels') and hasattr(self.app.control_panels, 'show_contours'):
            return self.app.control_panels.show_contours.get()
        return False
    
    def _get_source_image(self) -> Optional[np.ndarray]:
        """
        Get the appropriate source image based on current settings.
        
        Returns:
            The source image to process, or None if not available.
        """
        use_object_detection = (hasattr(self.app, 'masked_image') and 
                              self.app.masked_image is not None)
        
        if use_object_detection:
            return self.app.masked_image
        
        # Make sure enhanced image is created when enhancement is enabled
        if self.app.use_enhancement.get() and not hasattr(self.app, 'enhanced_image'):
            self.app.image_enhancer.apply_enhancement()
            return None  # Return early to avoid recursive loop
        
        # Use enhanced image if enhancement is enabled
        return (self.app.enhanced_image if self.app.use_enhancement.get() 
                else self.app.original_image)
    
    @performance_timer("image_preprocessing")
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale and apply blur preprocessing.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur
        blur_size = self.app.blur_size.get()
        if blur_size > 0:
            kernel_size = blur_size * 2 + 1  # Make sure it's odd
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        return gray
    
    def _apply_edge_algorithm(self, edge_method: str, gray: np.ndarray) -> np.ndarray:
        """
        Apply the selected edge detection algorithm.
        
        Args:
            edge_method: Name of the edge detection method
            gray: Preprocessed grayscale image
            
        Returns:
            Edge detection result
        """
        # Use the algorithm from the dictionary or fallback to Canny
        algorithm_func = self._edge_algorithms.get(edge_method, self._apply_canny)
        return algorithm_func(gray)
    
    @performance_timer("canny_edge_detection")
    def _apply_canny(self, gray: np.ndarray) -> np.ndarray: # it applies the Canny Edge Detection
        """Apply Canny edge detection algorithm."""
        ksize = self.app.apertureSize.get() # gets the aperturesize to apply Canny edge detection
        threshold1 = self.app.threshold1.get() # gets the higher and lower frequencies
        threshold2 = self.app.threshold2.get()
        
        return cv2.Canny(gray, threshold1, threshold2, apertureSize=ksize) # applying Canny edge Detection on the gray scale image
    
    @performance_timer("sobel_edge_detection")
    def _apply_sobel(self, gray: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection algorithm."""
        ksize = self.app.apertureSize.get()
        threshold = self.app.threshold1.get()
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)  # type: ignore
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)  # type: ignore
        
        sobel = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.convertScaleAbs(sobel)
        _, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        return edges
    
    @performance_timer("laplacian_edge_detection")
    def _apply_laplacian(self, gray: np.ndarray) -> np.ndarray:
        """Apply Laplacian edge detection algorithm."""
        ksize = self.app.apertureSize.get()
        threshold = self.app.threshold1.get()
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)  # type: ignore
        edges = cv2.convertScaleAbs(laplacian)
        _, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        return edges
    
    @performance_timer("prewitt_edge_detection")
    def _apply_prewitt(self, gray: np.ndarray) -> np.ndarray:
        """Apply Prewitt edge detection algorithm."""
        threshold = self.app.threshold1.get()
        
        kernelx = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
        kernely = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
        prewitt_x = cv2.filter2D(gray, -1, kernelx)
        prewitt_y = cv2.filter2D(gray, -1, kernely)
        
        prewitt = cv2.magnitude(prewitt_x.astype(np.float32), prewitt_y.astype(np.float32))
        edges = cv2.convertScaleAbs(prewitt)
        _, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        return edges
    
    @performance_timer("scharr_edge_detection")
    def _apply_scharr(self, gray: np.ndarray) -> np.ndarray:
        """Apply Scharr edge detection algorithm."""
        threshold = self.app.threshold1.get()
        
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)  # type: ignore
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)  # type: ignore
        
        scharr = cv2.magnitude(scharr_x, scharr_y)
        edges = cv2.convertScaleAbs(scharr)
        _, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        return edges
    
    @performance_timer("roberts_edge_detection")
    def _apply_roberts(self, gray: np.ndarray) -> np.ndarray:
        """Apply Roberts edge detection algorithm."""
        threshold = self.app.threshold1.get()
        
        # Roberts cross operator
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        # Apply kernels
        gx = cv2.filter2D(gray, cv2.CV_32F, roberts_x)  # type: ignore
        gy = cv2.filter2D(gray, cv2.CV_32F, roberts_y)  # type: ignore
        
        # Calculate magnitude
        roberts = cv2.magnitude(gx, gy)
        edges = cv2.convertScaleAbs(roberts)
        _, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        return edges
    
    @performance_timer("edge_post_processing")
    def _post_process_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Post-process the edges (invert, adjust thickness).
        
        Args:
            edges: Input edge image
            
        Returns:
            Post-processed edge image
        """
        # Invert edges if selected
        if self.app.invert_edges.get():
            edges = cv2.bitwise_not(edges)
            
        # Adjust edge thickness
        thickness = self.app.edge_thickness.get() # the thickness of the edges are taken from the app.py
        if thickness > 1:
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    @lru_cache(maxsize=8)
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]: # this function converts the frame to BGR to apply edge detection
        """
        Convert hex color to BGR (with caching for performance).
        
        Args:
            hex_color: Hex color string (e.g., "#FF0000")
            
        Returns:
            BGR color tuple
        """
        hex_color = hex_color.lstrip('#') # this will remove the '#' symbol from the color code ex: #AAA2348
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb_color[2], rgb_color[1], rgb_color[0])  # RGB to BGR
    
    @performance_timer("edge_visualization")
    def _apply_visualization(self, edges: np.ndarray, source_img: np.ndarray) -> None:
        """
        Apply visualization effects (color overlay, contours).
        
        Args:
            edges: Edge detection result
            source_img: Source image for visualization
        """
        # contour mean the outline of a shape
        # Check if contour detection is actually enabled via the checkbox
        contour_enabled = False
        if hasattr(self.app, 'control_panels') and hasattr(self.app.control_panels, 'show_contours'):
            contour_enabled = self.app.control_panels.show_contours.get() # calls the function show_contours from the file panels.py
        
        # Check if color selection is being used - HARD CODED FOR COLOR MODE
        color_selection_active = (hasattr(self.app, 'masked_image') and 
                                self.app.masked_image is not None) # the masked_image is set as None
        
        # Only apply contour mode if the checkbox is enabled AND settings are valid
        if contour_enabled:
            contour_type = self.app.contour_type.get()
            contour_display_mode = self.app.contour_display_mode.get()
            contour_source = self.app.contour_source.get()
            area_filter_enabled = self.app.area_filter_enabled.get()
            
            # Special handling when color selection is NOT used but contour source is edges + area filter
            if (not color_selection_active and 
                contour_source == "edges" and 
                area_filter_enabled):
                log_debug("Color selection not used, but edge detection + area filter enabled - using optimized path")
                # Ensure we have valid edge detection output for contour detection
                if edges is None or edges.size == 0:
                    log_warning("Invalid edge detection output for contour detection")
                    # Fallback to regular edge detection display
                    if self.app.color_overlay.get():
                        self._apply_color_overlay(edges, source_img)
                    else:
                        self.app.processed_image = edges
                    return
            
            if contour_type in ["external", "internal"] and contour_display_mode in ["over_original", "just_contours"]:
                # When contour mode is enabled, only show contours (no edge detection)
                self._apply_contour_mode(edges, source_img)
            else:
                # Invalid contour settings, fall back to edge detection
                if self.app.color_overlay.get():
                    self._apply_color_overlay(edges, source_img)
                else:
                    self.app.processed_image = edges
        elif self.app.color_overlay.get():
            # Apply color overlay if enabled and contour mode is disabled
            self._apply_color_overlay(edges, source_img)
        else:
            # Default: Use the edge image as the result
            self.app.processed_image = edges
    
    def _apply_color_overlay(self, edges: np.ndarray, source_img: np.ndarray) -> None: # this function adds a colored edge and merges it to the original one
        """Apply color overlay to edges."""
        # Get BGR color from hex
        bgr_color = self._hex_to_bgr(self.app.edge_color.get())
        
        # Create a colored version of the edges
        colored_edges = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for i in range(3):  # BGR channels
            colored_edges[:, :, i] = cv2.bitwise_and(
                np.full_like(edges, bgr_color[i]), edges
            )
        
        # Combine with original image
        if len(source_img.shape) == 3:
            # Color source image
            self.app.processed_image = cv2.addWeighted(source_img, 0.7, colored_edges, 0.3, 0) # this is like opacity
        else:
            # Grayscale source image - convert to BGR first
            source_bgr = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR)
            self.app.processed_image = cv2.addWeighted(source_bgr, 0.7, colored_edges, 0.3, 0)
    
    def _filter_contours_by_area(self, contours) -> List:
        """
        Filter contours by area with enhanced filtering options.
        
        This method maintains the original image size and does not compress or resize
        the image. Instead, it uses adaptive limits and timeouts based on image size
        to ensure efficient processing without quality loss.
        
        For large images without color selection, it uses stage-by-stage processing
        to prevent system crashes and improve performance.
        
        Args:
            contours: List of contours to filter
            
        Returns:
            Filtered list of contours
        """
        if not contours:
            log_info("No contours to filter - returning empty list")
            return []
        
        # Show loader for area filtering
        self._show_contour_loader_safe("Filtering contours by area...")
        
        try:
            import time
            start_time = time.time()
            
            log_info(f"Starting contour area filtering: {len(contours)} contours")
            
            # Get area filtering parameters with safety checks
            area_filter_enabled = self.app.area_filter_enabled.get()
            min_area = self.app.min_contour_area.get()
            max_area = self.app.max_contour_area.get()
            sort_mode = self.app.area_sort_mode.get()
            
            log_info(f"Filtering parameters: enabled={area_filter_enabled}, min_area={min_area}, max_area={max_area}, sort_mode={sort_mode}")
            
            # Check if this is edge detection + area filter combination
            contour_source = self.app.contour_source.get()
            is_edge_area_combination = (contour_source == "edges" and area_filter_enabled)
            
            # Check if color selection is being used
            color_selection_active = (hasattr(self.app, 'masked_image') and 
                                    self.app.masked_image is not None)
            
            log_info(f"Processing context: contour_source={contour_source}, edge+area={is_edge_area_combination}, color_selection={color_selection_active}")
            
            # Determine if we need stage-by-stage processing
            # Get the actual image size from the source image, not from contours
            if hasattr(self.app, 'original_image') and self.app.original_image is not None:
                actual_image_size = self.app.original_image.shape[0] * self.app.original_image.shape[1]
            elif hasattr(self.app, 'processed_image') and self.app.processed_image is not None:
                actual_image_size = self.app.processed_image.shape[0] * self.app.processed_image.shape[1]
            else:
                # Fallback to contour-based calculation if no image available
                actual_image_size = contours[0].shape[0] * contours[0].shape[1] if contours else 0
            
            needs_staged_processing = (is_edge_area_combination and 
                                     not color_selection_active and 
                                     (actual_image_size > 2000000 or len(contours) > 2000))
            
            log_info(f"Actual image size: {actual_image_size:,} pixels, contour count: {len(contours)}")
            log_info(f"Staged processing criteria: edge+area={is_edge_area_combination}, no_color_selection={not color_selection_active}, large_image={actual_image_size > 2000000}, many_contours={len(contours) > 2000}")
            
            if needs_staged_processing:
                log_info(f"Large image detected ({actual_image_size:,} pixels, {len(contours)} contours) - using stage-by-stage processing")
                # Update status to inform user about staged processing
                self.app.status_var.set("Processing large image with staged contour detection...")
                # Show loader for staged processing
                self._show_contour_loader_safe("Processing large image (staged)...")
                result = self._staged_contour_filtering(contours, min_area, max_area, sort_mode, start_time)
                # Hide loader when staged processing is complete
                self._hide_contour_loader_safe()
                return result
            else:
                log_info(f"Using regular contour filtering: image_size={actual_image_size:,}, contours={len(contours)}, edge+area={is_edge_area_combination}, color_selection={color_selection_active}")
                # Regular processing for smaller images or when color selection is active
                result = self._regular_contour_filtering(contours, min_area, max_area, sort_mode, 
                                                     is_edge_area_combination, color_selection_active, start_time)
                # Hide loader when area filtering is complete
                self._hide_contour_loader_safe()
                return result
            
        except Exception as e:
            log_error("Error in area filtering, returning all contours", exception=e)
            log_error(f"Fallback: returning all {len(contours)} contours due to error")
            # Hide loader on error
            self._hide_contour_loader_safe()
            # Return all contours as fallback
            return contours
    
    def _staged_contour_filtering(self, contours: List, min_area: int, max_area: int, 
                                 sort_mode: str, start_time: float) -> List:
        """
        Process contours in stages to prevent system crashes on large images.
        
        Args:
            contours: List of contours to filter
            min_area: Minimum contour area
            max_area: Maximum contour area
            sort_mode: Sorting mode
            start_time: Start time for timeout calculation
            
        Returns:
            Filtered list of contours
        """
        try:
            import time
            
            log_info(f"Starting staged contour filtering: {len(contours)} contours, area range {min_area}-{max_area}, sort={sort_mode}")
            
            # Stage 1: Initial contour limit based on image size
            # Get the actual image size from the source image, not from contours
            if hasattr(self.app, 'original_image') and self.app.original_image is not None:
                actual_image_size = self.app.original_image.shape[0] * self.app.original_image.shape[1]
            elif hasattr(self.app, 'processed_image') and self.app.processed_image is not None:
                actual_image_size = self.app.processed_image.shape[0] * self.app.processed_image.shape[1]
            else:
                # Fallback to contour-based calculation if no image available
                actual_image_size = contours[0].shape[0] * contours[0].shape[1] if contours else 0
            
            if actual_image_size > 5000000:  # 5MP+
                stage1_limit = 1500
                batch_size = 200
                log_info(f"Large image detected ({actual_image_size:,} pixels) - using conservative limits: {stage1_limit} contours, {batch_size} batch size")
            elif actual_image_size > 2000000:  # 2MP+
                stage1_limit = 2000
                batch_size = 300
                log_info(f"Medium image detected ({actual_image_size:,} pixels) - using moderate limits: {stage1_limit} contours, {batch_size} batch size")
            else:
                stage1_limit = 3000
                batch_size = 500
                log_info(f"Small image detected ({actual_image_size:,} pixels) - using standard limits: {stage1_limit} contours, {batch_size} batch size")
            
            # Update loader message for stage 1
            self._show_contour_loader_safe("Stage 1: Limiting contours...")
            
            if len(contours) > stage1_limit:
                log_info(f"Stage 1: Limiting contours from {len(contours)} to {stage1_limit} (image size: {actual_image_size:,} pixels)")
                self.app.status_var.set(f"Stage 1: Limiting contours from {len(contours)} to {stage1_limit}")
                contours = contours[:stage1_limit]
            else:
                log_info(f"Stage 1: No contour limiting needed ({len(contours)} <= {stage1_limit})")
            
            # Stage 2: Batch area calculation with progress updates
            log_info(f"Stage 2: Calculating areas for {len(contours)} contours in batches of {batch_size}")
            self.app.status_var.set(f"Stage 2: Calculating areas for {len(contours)} contours...")
            self._show_contour_loader_safe("Stage 2: Calculating areas...")
            contour_areas = []
            total_batches = (len(contours) + batch_size - 1) // batch_size
            log_info(f"Stage 2: Processing {total_batches} batches")
            
            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(contours))
                batch_contours = contours[batch_start:batch_end]
                
                # Check timeout for each batch
                if time.time() - start_time > 3.0:  # 3 second timeout per stage
                    log_warning(f"Stage 2 timeout after {batch_idx + 1}/{total_batches} batches (elapsed: {time.time() - start_time:.2f}s)")
                    break
                
                # Process batch
                batch_start_time = time.time()
                for i, contour in enumerate(batch_contours):
                    area = cv2.contourArea(contour)
                    contour_areas.append((batch_start + i, area))
                batch_time = time.time() - batch_start_time
                
                # Update progress
                if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    log_debug(f"Stage 2 progress: {progress:.1f}% ({batch_idx + 1}/{total_batches} batches) - batch {batch_idx + 1} took {batch_time:.3f}s")
                    self.app.status_var.set(f"Stage 2: {progress:.1f}% complete ({batch_idx + 1}/{total_batches} batches)")
            
            log_info(f"Stage 2 completed: calculated areas for {len(contour_areas)} contours")
            
            # Stage 3: Area filtering
            if max_area == 0:
                log_info(f"Stage 3: Filtering contours by area (min: {min_area}, no upper limit)")
                self.app.status_var.set(f"Stage 3: Filtering contours by area...")
                self._show_contour_loader_safe("Stage 3: Filtering contours...")
                filtered_indices = []
                stage3_start = time.time()
                
                for i, area in contour_areas:
                    if min_area <= area:
                        filtered_indices.append(i)
            else:
                log_info(f"Stage 3: Filtering contours by area range ({min_area} - {max_area})")
                self.app.status_var.set(f"Stage 3: Filtering contours by area...")
                self._show_contour_loader_safe("Stage 3: Filtering contours...")
                filtered_indices = []
                stage3_start = time.time()
                
                for i, area in contour_areas:
                    if min_area <= area <= max_area:
                        filtered_indices.append(i)
            
            stage3_time = time.time() - stage3_start
            log_info(f"Stage 3 completed in {stage3_time:.3f}s: filtered to {len(filtered_indices)} contours (from {len(contour_areas)})")
            self.app.status_var.set(f"Stage 3: Filtered to {len(filtered_indices)} contours")
            
            # Stage 4: Sorting (if enabled and within timeout)
            if sort_mode != "none" and time.time() - start_time < 2.0:  # 2 second timeout for sorting
                log_info(f"Stage 4: Sorting {len(filtered_indices)} contours by {sort_mode}")
                self.app.status_var.set(f"Stage 4: Sorting contours by {sort_mode}...")
                self._show_contour_loader_safe("Stage 4: Sorting contours...")
                stage4_start = time.time()
                
                filtered_areas = [(i, contour_areas[i][1]) for i in filtered_indices]
                
                if sort_mode == "ascending":
                    filtered_indices = [i for i, _ in sorted(filtered_areas, key=lambda x: x[1])]
                    log_debug(f"Stage 4: Sorted {len(filtered_indices)} contours in ascending order")
                elif sort_mode == "descending":
                    filtered_indices = [i for i, _ in sorted(filtered_areas, key=lambda x: x[1], reverse=True)]
                    log_debug(f"Stage 4: Sorted {len(filtered_indices)} contours in descending order")
                
                stage4_time = time.time() - stage4_start
                log_info(f"Stage 4 completed in {stage4_time:.3f}s: sorting finished")
                self.app.status_var.set(f"Stage 4: Sorting completed")
            else:
                if sort_mode != "none":
                    log_warning(f"Stage 4 skipped: timeout exceeded ({time.time() - start_time:.2f}s > 2.0s) or no sorting needed")
                else:
                    log_info(f"Stage 4 skipped: no sorting requested (sort_mode = {sort_mode})")
            
            processing_time = time.time() - start_time
            log_info(f"Staged contour filtering completed in {processing_time:.3f}s: {len(filtered_indices)} contours from {len(contours)} original")
            log_info(f"Performance summary: Stage1={time.time() - start_time - processing_time:.3f}s, Stage2={batch_time:.3f}s, Stage3={stage3_time:.3f}s, Stage4={stage4_time if sort_mode != 'none' else 0:.3f}s")
            self.app.status_var.set(f"Contour detection completed: {len(filtered_indices)} contours in {processing_time:.1f}s")
            
            # Hide loader when staged processing is complete
            self._hide_contour_loader_safe()
            
            return [contours[i] for i in filtered_indices]
            
        except Exception as e:
            log_error("Error in staged contour filtering", exception=e)
            log_error(f"Falling back to first 1000 contours due to error")
            # Hide loader on error
            self._hide_contour_loader_safe()
            return contours[:1000]  # Return first 1000 contours as fallback
    
    def _regular_contour_filtering(self, contours: List, min_area: int, max_area: int, 
                                  sort_mode: str, is_edge_area_combination: bool, 
                                  color_selection_active: bool, start_time: float) -> List:
        """
        Regular contour filtering for smaller images or when color selection is active.
        
        Args:
            contours: List of contours to filter
            min_area: Minimum contour area
            max_area: Maximum contour area
            sort_mode: Sorting mode
            is_edge_area_combination: Whether this is edge+area combination
            color_selection_active: Whether color selection is active
            start_time: Start time for timeout calculation
            
        Returns:
            Filtered list of contours
        """
        try:
            import time
            
            log_info(f"Starting regular contour filtering: {len(contours)} contours, area range {min_area}-{max_area}, sort={sort_mode}")
            log_info(f"Processing mode: edge+area={is_edge_area_combination}, color_selection={color_selection_active}")
            
            # Show loader for regular processing
            self._show_contour_loader_safe("Processing contours...")
            
            if is_edge_area_combination:
                if not color_selection_active:
                    log_info("Color selection not used + edge detection + area filter - applying stricter limits")
                    # Use adaptive limits based on image size instead of fixed limits
                    image_size = contours[0].shape[0] * contours[0].shape[1] if contours else 0
                    if image_size > 5000000:  # 5MP+
                        contour_limit = 1000
                        log_info(f"Large image ({image_size:,} pixels) - using strict limit: {contour_limit} contours")
                    elif image_size > 2000000:  # 2MP+
                        contour_limit = 1500
                        log_info(f"Medium image ({image_size:,} pixels) - using moderate limit: {contour_limit} contours")
                    else:
                        contour_limit = 2000
                        log_info(f"Small image ({image_size:,} pixels) - using standard limit: {contour_limit} contours")
                else:
                    log_info("Edge detection + area filter combination - applying strict limits")
                    image_size = contours[0].shape[0] * contours[0].shape[1] if contours else 0
                    if image_size > 5000000:  # 5MP+
                        contour_limit = 1500
                        log_info(f"Large image ({image_size:,} pixels) - using limit: {contour_limit} contours")
                    elif image_size > 2000000:  # 2MP+
                        contour_limit = 2000
                        log_info(f"Medium image ({image_size:,} pixels) - using limit: {contour_limit} contours")
                    else:
                        contour_limit = 3000
                        log_info(f"Small image ({image_size:,} pixels) - using limit: {contour_limit} contours")
                
                if len(contours) > contour_limit:
                    log_warning(f"Too many contours for edge+area combination ({len(contours)}), limiting to {contour_limit}")
                    contours = contours[:contour_limit]
                else:
                    log_info(f"Contour count ({len(contours)}) within limit ({contour_limit}) - no limiting needed")
            
            # Validate area parameters
            if min_area < 0:
                min_area = 0
                log_warning("Minimum area was negative, setting to 0")
            if max_area < min_area:
                max_area = min_area + 1000  # Set a reasonable default
                log_warning(f"Maximum area was less than minimum, setting to {max_area}")
            
            log_info(f"Area filtering parameters: min={min_area}, max={max_area}")
            
            # Limit number of contours to prevent excessive processing
            # Use adaptive limits based on image size
            # Get the actual image size from the source image, not from contours
            if hasattr(self.app, 'original_image') and self.app.original_image is not None:
                actual_image_size = self.app.original_image.shape[0] * self.app.original_image.shape[1]
            elif hasattr(self.app, 'processed_image') and self.app.processed_image is not None:
                actual_image_size = self.app.processed_image.shape[0] * self.app.processed_image.shape[1]
            else:
                # Fallback to contour-based calculation if no image available
                actual_image_size = contours[0].shape[0] * contours[0].shape[1] if contours else 0
            
            if actual_image_size > 5000000:  # 5MP+
                max_contours = 3000
                log_info(f"Large image ({actual_image_size:,} pixels) - max contours: {max_contours}")
            elif actual_image_size > 2000000:  # 2MP+
                max_contours = 5000
                log_info(f"Medium image ({actual_image_size:,} pixels) - max contours: {max_contours}")
            else:
                max_contours = 10000
                log_info(f"Small image ({actual_image_size:,} pixels) - max contours: {max_contours}")
                
            if len(contours) > max_contours:
                log_warning(f"Too many contours ({len(contours)}), limiting to {max_contours}")
                contours = contours[:max_contours]
            else:
                log_info(f"Contour count ({len(contours)}) within max limit ({max_contours})")
            
            # Calculate areas for all contours with timeout protection
            log_info(f"Starting area calculation for {len(contours)} contours")
            contour_areas = []
            # Use adaptive timeout based on image size
            # Use the actual_image_size we calculated earlier
            if actual_image_size > 5000000:  # 5MP+
                base_timeout = 4.0
                log_info(f"Large image ({actual_image_size:,} pixels) - base timeout: {base_timeout}s")
            elif actual_image_size > 2000000:  # 2MP+
                base_timeout = 3.0
                log_info(f"Medium image ({actual_image_size:,} pixels) - base timeout: {base_timeout}s")
            else:
                base_timeout = 2.0
                log_info(f"Small image ({actual_image_size:,} pixels) - base timeout: {base_timeout}s")
                
            if is_edge_area_combination:
                if not color_selection_active:
                    timeout_limit = base_timeout * 0.75  # Shorter timeout when color selection not used
                    log_info(f"Edge+area combination without color selection - timeout: {timeout_limit}s")
                else:
                    timeout_limit = base_timeout  # Regular timeout for edge+area combination
                    log_info(f"Edge+area combination with color selection - timeout: {timeout_limit}s")
            else:
                timeout_limit = base_timeout * 1.5  # Longer timeout for regular processing
                log_info(f"Regular processing - timeout: {timeout_limit}s")
            
            area_calc_start = time.time()
            for i, contour in enumerate(contours):
                if time.time() - start_time > timeout_limit:
                    log_warning(f"Area calculation timeout ({timeout_limit}s), stopping early at contour {i}/{len(contours)}")
                    break
                area = cv2.contourArea(contour)
                contour_areas.append((i, area))
            
            area_calc_time = time.time() - area_calc_start
            log_info(f"Area calculation completed in {area_calc_time:.3f}s: {len(contour_areas)} areas calculated")
            
            # Apply area filtering if enabled
            area_filter_enabled = self.app.area_filter_enabled.get()
            if area_filter_enabled:
                # Handle no limit case (max_area = 0)
                if max_area == 0:
                    log_info(f"Applying area filtering: {min_area} <= area (no upper limit)")
                    filtering_start = time.time()
                    filtered_indices = []
                    for i, area in contour_areas:
                        if min_area <= area:
                            filtered_indices.append(i)
                    filtering_time = time.time() - filtering_start
                    log_info(f"Area filtering completed in {filtering_time:.3f}s: {len(filtered_indices)} contours passed filter (from {len(contour_areas)})")
                else:
                    log_info(f"Applying area filtering: {min_area} <= area <= {max_area}")
                    filtering_start = time.time()
                    filtered_indices = []
                    for i, area in contour_areas:
                        if min_area <= area <= max_area:
                            filtered_indices.append(i)
                    filtering_time = time.time() - filtering_start
                    log_info(f"Area filtering completed in {filtering_time:.3f}s: {len(filtered_indices)} contours passed filter (from {len(contour_areas)})")
            else:
                log_info("Area filtering disabled - including all contours")
                # If filtering is disabled, include all contours
                filtered_indices = [i for i, _ in contour_areas]
            
            # Apply sorting if specified (with timeout protection)
            # Use adaptive timeout based on image size
            if actual_image_size > 5000000:  # 5MP+
                base_sort_timeout = 5.0
                log_info(f"Large image ({actual_image_size:,} pixels) - base sort timeout: {base_sort_timeout}s")
            elif actual_image_size > 2000000:  # 2MP+
                base_sort_timeout = 4.0
                log_info(f"Medium image ({actual_image_size:,} pixels) - base sort timeout: {base_sort_timeout}s")
            else:
                base_sort_timeout = 3.0
                log_info(f"Small image ({actual_image_size:,} pixels) - base sort timeout: {base_sort_timeout}s")
                
            if is_edge_area_combination:
                if not color_selection_active:
                    sort_timeout = base_sort_timeout * 0.75  # Shorter timeout when color selection not used
                    log_info(f"Edge+area without color selection - sort timeout: {sort_timeout}s")
                else:
                    sort_timeout = base_sort_timeout  # Regular timeout for edge+area combination
                    log_info(f"Edge+area with color selection - sort timeout: {sort_timeout}s")
            else:
                sort_timeout = base_sort_timeout * 1.5  # Longer timeout for regular processing
                log_info(f"Regular processing - sort timeout: {sort_timeout}s")
                
            if sort_mode != "none" and time.time() - start_time < sort_timeout:
                log_info(f"Applying sorting: {sort_mode} order for {len(filtered_indices)} contours")
                sorting_start = time.time()
                # Get areas for filtered contours
                filtered_areas = [(i, contour_areas[i][1]) for i in filtered_indices]
                
                if sort_mode == "ascending":
                    filtered_indices = [i for i, _ in sorted(filtered_areas, key=lambda x: x[1])]
                    log_debug(f"Sorted {len(filtered_indices)} contours in ascending order")
                elif sort_mode == "descending":
                    filtered_indices = [i for i, _ in sorted(filtered_areas, key=lambda x: x[1], reverse=True)]
                    log_debug(f"Sorted {len(filtered_indices)} contours in descending order")
                
                sorting_time = time.time() - sorting_start
                log_info(f"Sorting completed in {sorting_time:.3f}s")
            else:
                if sort_mode != "none":
                    log_warning(f"Sorting skipped: timeout exceeded ({time.time() - start_time:.2f}s > {sort_timeout}s)")
                else:
                    log_info(f"Sorting skipped: no sorting requested (sort_mode = {sort_mode})")
            
            processing_time = time.time() - start_time
            log_info(f"Regular contour filtering completed in {processing_time:.3f}s: {len(filtered_indices)} contours from {len(contours)} original")
            log_info(f"Performance breakdown: area_calc={area_calc_time:.3f}s, filtering={filtering_time if area_filter_enabled else 0:.3f}s, sorting={sorting_time if sort_mode != 'none' else 0:.3f}s")
            self.app.status_var.set(f"Contour detection completed: {len(filtered_indices)} contours in {processing_time:.1f}s")
            
            # Hide loader when regular processing is complete
            self._hide_contour_loader_safe()
            
            # Return filtered and sorted contours
            return [contours[i] for i in filtered_indices]
            
        except Exception as e:
            log_error("Error in regular contour filtering", exception=e)
            # Hide loader on error
            self._hide_contour_loader_safe()
            return contours
    
    def _apply_contour_mode(self, edges: np.ndarray, source_img: np.ndarray) -> None:
        """
        Apply contour detection and drawing mode.
        
        This method maintains the original image size and does not compress or resize
        the image, ensuring high-quality contour detection results.
        """
        try:
            log_debug("Starting contour detection mode")
            
            # Show loader for contour detection
            self._show_contour_loader_safe("Detecting contours...")
            
            contour_type = self.app.contour_type.get()
            contour_display_mode = self.app.contour_display_mode.get()
            contour_source = self.app.contour_source.get()
            area_filter_enabled = self.app.area_filter_enabled.get()
            
            # Check if color selection is being used
            color_selection_active = (hasattr(self.app, 'masked_image') and 
                                    self.app.masked_image is not None)
            
            # Log when both edge detection and area filter are enabled
            if contour_source == "edges" and area_filter_enabled:
                if not color_selection_active:
                    log_info("Color selection not used + edge detection + area filter combination - applying performance optimizations")
                else:
                    log_info("Contour source set to 'edges' and area filter enabled: this combination may impact performance.")
            
            log_debug(f"Contour settings: type={contour_type}, display={contour_display_mode}, source={contour_source}")
            
            # Determine contour retrieval mode based on contour type
            if contour_type == "external":
                retrieval_mode = cv2.RETR_EXTERNAL
            elif contour_type == "internal":
                retrieval_mode = cv2.RETR_TREE
            else:  # Default to external
                retrieval_mode = cv2.RETR_EXTERNAL
            
            # Choose the input image for contour detection
            if contour_source == "direct":
                # Use the source image directly for contour detection
                contour_input = source_img
                # Convert to grayscale if it's a color image
                if len(contour_input.shape) == 3:
                    contour_input = cv2.cvtColor(contour_input, cv2.COLOR_BGR2GRAY)
                log_debug("Using direct image for contour detection")
            else:
                # Use edge detection results (default behavior)
                contour_input = edges
                log_debug("Using edge detection results for contour detection")
                
                # Validate edge detection output before proceeding
                if contour_input is None or contour_input.size == 0:
                    log_error("Invalid edge detection output for contour detection")
                    # Fallback to regular edge detection display
                    if self.app.color_overlay.get():
                        self._apply_color_overlay(edges, source_img)
                    else:
                        self.app.processed_image = edges
                    return
                
                # Additional safety check for edge detection + area filter combination
                if area_filter_enabled:
                    if not color_selection_active:
                        log_debug("Color selection not used + edge + area filter - applying enhanced safeguards")
                    else:
                        log_debug("Edge detection + area filter combination detected - applying extra safeguards")
                    
                    # Instead of resizing, we'll use more efficient contour detection
                    # and apply stricter limits in the area filtering
            
            # Validate input image before contour detection
            if contour_input is None or contour_input.size == 0:
                log_error("Invalid contour input image")
                return
            
            # Check image dimensions and log info (but don't resize)
            image_size = contour_input.shape[0] * contour_input.shape[1]
            if image_size > 10000000:  # 10MP limit
                log_info(f"Large image detected ({image_size:,} pixels) - using optimized contour detection")
            else:
                log_debug(f"Image size: {image_size:,} pixels")
            
            log_debug(f"Finding contours with input shape: {contour_input.shape}")
            
            # Find contours in the selected input image with timeout protection
            import time
            start_time = time.time()
            contours, _ = cv2.findContours(contour_input, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
            contour_time = time.time() - start_time
            
            # Use adaptive timeout based on image size
            image_size = contour_input.shape[0] * contour_input.shape[1]
            if image_size > 5000000:  # 5MP+
                max_contour_time = 8.0
            elif image_size > 2000000:  # 2MP+
                max_contour_time = 6.0
            else:
                max_contour_time = 5.0
                
            if contour_time > max_contour_time:
                log_warning(f"Contour detection took {contour_time:.2f}s, limiting contours")
                # Limit number of contours to prevent freezing
                if len(contours) > 1000:
                    contours = contours[:1000]
                    log_warning(f"Limited contours to 1000 (was {len(contours)})")
            
            log_debug(f"Found {len(contours)} contours in {contour_time:.3f}s")
            
            # Enhanced area filtering
            filtered_contours = self._filter_contours_by_area(contours)
            
            # Ensure we have valid contour data even if no contours are found
            if not contours:
                contours = []
            if not filtered_contours:
                filtered_contours = []
            
            # CRITICAL FIX: Initialize contour visibility with proper error handling
            # This prevents UI crashes when color selection is disabled
            try:
                # Initialize contour visibility if it doesn't exist
                if not hasattr(self.app, 'contour_visibility'):
                    self.app.contour_visibility = []
                
                # Improved contour visibility management with better error handling
                # Only reset visibility if the number of contours has changed significantly
                # This preserves user's visibility preferences when possible
                current_visibility_count = len(self.app.contour_visibility)
                new_contour_count = len(filtered_contours)
                
                if current_visibility_count == 0:
                    # First time - initialize all as visible
                    self.app.contour_visibility = [True] * new_contour_count
                    log_debug(f"Initialized contour visibility for {new_contour_count} contours")
                elif abs(current_visibility_count - new_contour_count) > 2:
                    # Significant change in contour count - reset visibility
                    self.app.contour_visibility = [True] * new_contour_count
                    log_debug(f"Reset contour visibility due to significant count change: {current_visibility_count} -> {new_contour_count}")
                else:
                    # Minor change - adjust visibility list size while preserving existing states
                    if new_contour_count > current_visibility_count:
                        # Add new contours as visible
                        additional_contours = new_contour_count - current_visibility_count
                        self.app.contour_visibility.extend([True] * additional_contours)
                        log_debug(f"Extended contour visibility by {additional_contours} contours")
                    elif new_contour_count < current_visibility_count:
                        # Remove excess visibility entries
                        self.app.contour_visibility = self.app.contour_visibility[:new_contour_count]
                        log_debug(f"Truncated contour visibility from {current_visibility_count} to {new_contour_count}")
                
                # CRITICAL: Validate visibility list length matches contour count
                if len(self.app.contour_visibility) != len(filtered_contours):
                    log_warning(f"Contour visibility mismatch: {len(self.app.contour_visibility)} vs {len(filtered_contours)} contours")
                    # Fix the mismatch by creating a new visibility list
                    self.app.contour_visibility = [True] * len(filtered_contours)
                    log_debug(f"Fixed contour visibility mismatch - reset to {len(filtered_contours)} contours")
                    
            except Exception as e:
                log_error("Error initializing contour visibility", exception=e)
                # Emergency fallback - create a safe visibility list
                self.app.contour_visibility = [True] * len(filtered_contours)
                log_debug(f"Emergency fallback: created visibility list for {len(filtered_contours)} contours")
            
            # Store contour information for UI display with proper error handling
            try:
                # Calculate contour properties with error handling
                areas = []
                perimeters = []
                point_counts = []
                total_points = 0
                filtered_points = 0
                
                # Calculate properties for all contours
                for contour in contours:
                    try:
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        points = len(contour)
                        total_points += points
                        
                        areas.append(area)
                        perimeters.append(perimeter)
                        point_counts.append(points)
                    except Exception as e:
                        log_error(f"Error calculating contour properties: {e}")
                        # Use fallback values
                        areas.append(0.0)
                        perimeters.append(0.0)
                        point_counts.append(0)
                
                # Calculate properties for filtered contours
                filtered_areas = []
                filtered_perimeters = []
                filtered_point_counts = []
                
                for contour in filtered_contours:
                    try:
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        points = len(contour)
                        filtered_points += points
                        
                        filtered_areas.append(area)
                        filtered_perimeters.append(perimeter)
                        filtered_point_counts.append(points)
                    except Exception as e:
                        log_error(f"Error calculating filtered contour properties: {e}")
                        # Use fallback values
                        filtered_areas.append(0.0)
                        filtered_perimeters.append(0.0)
                        filtered_point_counts.append(0)
                
                # Create complete contour info structure
                self.app.contour_info = {
                    'total_contours': len(contours),
                    'filtered_contours': len(filtered_contours),
                    'total_points': total_points,
                    'filtered_points': filtered_points,
                    'areas': filtered_areas,
                    'perimeters': filtered_perimeters,
                    'point_counts': filtered_point_counts,
                    'timestamp': time.time()  # Add timestamp for debugging
                }
                
                log_debug(f"Contour info stored: {len(filtered_contours)} contours, {len(filtered_point_counts)} point counts")
                
            except Exception as e:
                log_error("Error storing contour info", exception=e)
                # Create minimal but valid contour info to prevent crashes
                self.app.contour_info = {
                    'total_contours': len(contours),
                    'filtered_contours': len(filtered_contours),
                    'total_points': 0,
                    'filtered_points': 0,
                    'areas': [],
                    'perimeters': [],
                    'point_counts': [],
                    'timestamp': time.time()
                }
            
            # Get BGR color from hex
            bgr_color = self._hex_to_bgr(self.app.edge_color.get())
            
            # CRITICAL FIX: Safe contour filtering with bounds checking
            visible_contours = []
            try:
                for i, contour in enumerate(filtered_contours):
                    # Add bounds checking to prevent index errors
                    if (hasattr(self.app, 'contour_visibility') and 
                        i < len(self.app.contour_visibility) and 
                        self.app.contour_visibility[i]):
                        visible_contours.append(contour)
                    elif not hasattr(self.app, 'contour_visibility'):
                        # If visibility list doesn't exist, show all contours
                        visible_contours.append(contour)
                        log_warning("Contour visibility list missing - showing all contours")
                    else:
                        # If index is out of bounds, show the contour anyway
                        visible_contours.append(contour)
                        log_warning(f"Contour visibility index {i} out of bounds - showing contour anyway")
            except Exception as e:
                log_error("Error filtering visible contours", exception=e)
                # Emergency fallback - show all contours
                visible_contours = filtered_contours
                log_debug("Emergency fallback: showing all contours due to filtering error")
            
            # Cache the filtered contours for visibility toggling
            self._last_filtered_contours = filtered_contours
            
            # Apply contour processing options (convex hull)
            processed_contours = self._apply_contour_processing(visible_contours)
            
            # Handle different contour display modes
            if contour_display_mode == "over_original":
                # Draw contours over the original image
                if len(source_img.shape) == 3:
                    result_img = source_img.copy()
                else:
                    result_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(result_img, processed_contours, -1, bgr_color, self.app.edge_thickness.get())
                
            elif contour_display_mode == "just_contours":
                # Show only contours on black background
                result_img = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype=np.uint8)
                cv2.drawContours(result_img, processed_contours, -1, bgr_color, self.app.edge_thickness.get())
                
            else:
                # Default behavior (over_original mode)
                if len(source_img.shape) == 3:
                    result_img = source_img.copy()
                else:
                    result_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(result_img, processed_contours, -1, bgr_color, self.app.edge_thickness.get())
            
            self.app.processed_image = result_img
            
            # Hide loader when contour processing is complete
            self._hide_contour_loader_safe()
            
        except Exception as e:
            log_error("Error in contour detection mode", exception=e)
            # Hide loader on error
            self._hide_contour_loader_safe()
            # Fallback to simple edge detection
            if self.app.color_overlay.get():
                self._apply_color_overlay(edges, source_img)
            else:
                self.app.processed_image = edges
            return
    
    @error_handler(error_message="Failed to update display")
    def _update_display(self) -> None:
        """Update the display with the processed image."""
        if hasattr(self.app, 'processed_image') and self.app.processed_image is not None:
            self.app.image_display.display_image(
                self.app.processed_image, 
                self.app.image_display.result_display
            )
    
    def _safe_update_ui_after_completion(self) -> None:
        """Safely update UI after edge detection completion to prevent crashes and stuck UI."""
        try:
            # Add a longer delay to prevent UI oscillation
            import time
            current_time = time.time()
            
            # Check if we're updating too frequently
            if hasattr(self, '_last_ui_update_time'):
                time_since_last = current_time - self._last_ui_update_time
                if time_since_last < 1.0:  # Increased minimum delay to 1 second
                    log_debug(f"UI update too frequent ({time_since_last:.3f}s), skipping")
                    return
            
            self._last_ui_update_time = current_time
            
            # CRITICAL FIX: Prevent multiple simultaneous UI updates
            if hasattr(self, '_ui_update_in_progress') and self._ui_update_in_progress:
                log_debug("UI update already in progress, skipping")
                return
            
            self._ui_update_in_progress = True
            
            # CRITICAL FIX: Add watchdog to detect and recover from UI stuck state
            def ui_watchdog():
                try:
                    # Check if UI update is taking too long
                    if hasattr(self, '_ui_update_start_time'):
                        elapsed = time.time() - self._ui_update_start_time
                        if elapsed > 5.0:  # 5 second timeout
                            log_warning(f"UI update taking too long ({elapsed:.1f}s), triggering recovery")
                            # Trigger UI recovery
                            if hasattr(self.app, 'recover_from_ui_stuck'):
                                self.app.root.after(0, self.app.recover_from_ui_stuck)
                            # Reset update flag
                            self._ui_update_in_progress = False
                            return
                except Exception as e:
                    log_error("Error in UI watchdog", exception=e)
            
            # Start watchdog timer
            self._ui_update_start_time = current_time
            if hasattr(self.app, 'root') and self.app.root:
                self.app.root.after(5000, ui_watchdog)  # 5 second watchdog
            
            # Update contour info display if contours are enabled
            if (hasattr(self.app, 'control_panels') and 
                hasattr(self.app.control_panels, 'show_contours') and 
                self.app.control_panels.show_contours.get()):
                
                # Use a longer delay to ensure all processing is complete
                # Also add a check to ensure contour data is ready
                def delayed_update():
                    try:
                        # CRITICAL FIX: Better validation of contour data
                        # Check if contour info exists and is properly structured
                        if (hasattr(self.app, 'contour_info') and 
                            self.app.contour_info and 
                            isinstance(self.app.contour_info, dict) and
                            'timestamp' in self.app.contour_info):
                            
                            # Check if data is recent (within last 10 seconds)
                            data_age = time.time() - self.app.contour_info['timestamp']
                            if data_age < 10.0:
                                # Additional validation for required keys
                                required_keys = ['total_contours', 'filtered_contours', 'areas', 'perimeters', 'point_counts']
                                if all(key in self.app.contour_info for key in required_keys):
                                    self._update_contour_info_safe()
                                else:
                                    log_warning("Contour info missing required keys, skipping UI update")
                                    # Try to create minimal valid contour info
                                    self._create_minimal_contour_info()
                            else:
                                log_warning(f"Contour data too old ({data_age:.1f}s), skipping UI update")
                        else:
                            log_warning("Contour data not ready or invalid, skipping UI update")
                            # Try to create minimal valid contour info if contours are enabled
                            if (hasattr(self.app, 'control_panels') and 
                                hasattr(self.app.control_panels, 'show_contours') and 
                                self.app.control_panels.show_contours.get()):
                                self._create_minimal_contour_info()
                    except Exception as e:
                        log_error("Error in delayed contour update", exception=e)
                        # Emergency fallback - try to create minimal contour info
                        try:
                            self._create_minimal_contour_info()
                        except Exception as fallback_error:
                            log_error("Emergency fallback failed", exception=fallback_error)
                    finally:
                        # CRITICAL: Reset the update flag
                        self._ui_update_in_progress = False
                        # Clear watchdog start time
                        if hasattr(self, '_ui_update_start_time'):
                            delattr(self, '_ui_update_start_time')
                
                # Increased delay to 500ms to prevent UI conflicts
                self.app.root.after(500, delayed_update)
            else:
                # No contour update needed, reset flag immediately
                self._ui_update_in_progress = False
                # Clear watchdog start time
                if hasattr(self, '_ui_update_start_time'):
                    delattr(self, '_ui_update_start_time')
            
            # Update status with a delay to prevent conflicts
            def update_status():
                try:
                    self.app.status_var.set("Edge detection completed successfully")
                    
                    # CRITICAL: Auto-optimize contour display for large counts
                    if hasattr(self.app, 'optimize_contour_display_for_large_counts'):
                        self.app.root.after(200, self.app.optimize_contour_display_for_large_counts)
                        
                except Exception as e:
                    log_error("Error updating status", exception=e)
            
            self.app.root.after(100, update_status)
            
        except Exception as e:
            log_error("Error in safe UI update", exception=e)
            # CRITICAL: Always reset the update flag on error
            if hasattr(self, '_ui_update_in_progress'):
                self._ui_update_in_progress = False
            # Clear watchdog start time
            if hasattr(self, '_ui_update_start_time'):
                delattr(self, '_ui_update_start_time')
    
    def _create_minimal_contour_info(self) -> None:
        """Create minimal but valid contour info to prevent UI crashes."""
        try:
            import time
            
            # Create minimal contour info structure
            self.app.contour_info = {
                'total_contours': 0,
                'filtered_contours': 0,
                'total_points': 0,
                'filtered_points': 0,
                'areas': [],
                'perimeters': [],
                'point_counts': [],
                'timestamp': time.time()
            }
            
            # Initialize empty contour visibility list
            if not hasattr(self.app, 'contour_visibility'):
                self.app.contour_visibility = []
            
            log_debug("Created minimal contour info structure")
            
            # Try to update the UI display
            if hasattr(self.app, 'control_panels') and hasattr(self.app.control_panels, '_update_contour_info_display'):
                self.app.control_panels._update_contour_info_display()
                
        except Exception as e:
            log_error("Error creating minimal contour info", exception=e)
    
    def _update_contour_info_safe(self) -> None:
        """Safely update contour info display with error handling."""
        try:
            # Validate contour data before updating UI
            self._validate_contour_data()
            
            if hasattr(self.app, 'control_panels') and hasattr(self.app.control_panels, '_update_contour_info_display'):
                self.app.control_panels._update_contour_info_display()
        except Exception as e:
            log_error("Error updating contour info display", exception=e)
    
    def _show_contour_loader_safe(self, message: str = "Processing contours...") -> None:
        """Safely show contour loader in main thread."""
        try:
            if hasattr(self.app, 'root') and self.app.root:
                self.app.root.after(0, lambda: self._show_contour_loader_ui(message))
        except Exception as e:
            log_error("Error showing contour loader", exception=e)
    
    def _hide_contour_loader_safe(self) -> None:
        """Safely hide contour loader in main thread."""
        try:
            if hasattr(self.app, 'root') and self.app.root:
                self.app.root.after(0, self._hide_contour_loader_ui)
        except Exception as e:
            log_error("Error hiding contour loader", exception=e)
    
    def _show_contour_loader_ui(self, message: str) -> None:
        """Show contour loader in UI (called from main thread)."""
        try:
            if hasattr(self.app, 'control_panels') and hasattr(self.app.control_panels, 'show_contour_loader'):
                self.app.control_panels.show_contour_loader(message)
        except Exception as e:
            log_error("Error showing contour loader UI", exception=e)
    
    def _hide_contour_loader_ui(self) -> None:
        """Hide contour loader in UI (called from main thread)."""
        try:
            if hasattr(self.app, 'control_panels') and hasattr(self.app.control_panels, 'hide_contour_loader'):
                self.app.control_panels.hide_contour_loader()
        except Exception as e:
            log_error("Error hiding contour loader UI", exception=e)
    
    def _validate_contour_data(self) -> None:
        """Validate and clean up contour data to prevent UI crashes."""
        try:
            if not hasattr(self.app, 'contour_info') or not self.app.contour_info:
                log_debug("No contour info to validate")
                return
            
            info = self.app.contour_info
            
            # Ensure all required keys exist
            required_keys = ['total_contours', 'filtered_contours', 'areas', 'perimeters', 'point_counts']
            for key in required_keys:
                if key not in info:
                    log_warning(f"Missing required key in contour info: {key}")
                    if key in ['areas', 'perimeters', 'point_counts']:
                        info[key] = []
                    elif key in ['total_contours', 'filtered_contours']:
                        info[key] = 0
            
            # Validate data consistency
            filtered_count = info.get('filtered_contours', 0)
            
            # Ensure arrays have correct length
            for key in ['areas', 'perimeters', 'point_counts']:
                if key in info:
                    array = info[key]
                    if len(array) != filtered_count:
                        log_warning(f"Array length mismatch for {key}: {len(array)} vs {filtered_count}")
                        if len(array) > filtered_count:
                            info[key] = array[:filtered_count]
                        else:
                            # Extend with default values
                            if key == 'areas':
                                info[key].extend([0.0] * (filtered_count - len(array)))
                            elif key == 'perimeters':
                                info[key].extend([0.0] * (filtered_count - len(array)))
                            else:  # point_counts
                                info[key].extend([0] * (filtered_count - len(array)))
            
            # Validate visibility list
            if hasattr(self.app, 'contour_visibility'):
                if len(self.app.contour_visibility) != filtered_count:
                    log_warning(f"Visibility list length mismatch: {len(self.app.contour_visibility)} vs {filtered_count}")
                    if len(self.app.contour_visibility) > filtered_count:
                        self.app.contour_visibility = self.app.contour_visibility[:filtered_count]
                    else:
                        self.app.contour_visibility.extend([True] * (filtered_count - len(self.app.contour_visibility)))
            
            log_debug(f"Contour data validation complete: {filtered_count} contours")
            
        except Exception as e:
            log_error("Error validating contour data", exception=e)
    
    def _apply_contour_processing(self, contours: List) -> List:
        """
        Apply contour processing options using the ContourProcessor.
        
        Args:
            contours: List of contours to process
            
        Returns:
            List of processed contours
        """
        return self.contour_processor.process_contours(contours) 