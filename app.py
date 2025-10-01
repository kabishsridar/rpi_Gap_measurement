import tkinter as tk # importing tkinter for the front end displays
from tkinter import ttk
from tkinter import filedialog, colorchooser, simpledialog, messagebox

import cv2 # importing the modules which are required
import numpy as np
import os
import sys
import threading
from typing import Optional, Dict, Any
import time # Added for time.time()
from datetime import datetime

# Import our modules (files)
from ui.image_display import ImageDisplay
from ui.panels import ControlPanels
from ui.toolbar import Toolbar
from ui.video_dialog import VideoPanoramaDialog
from ui.settings_editor import SettingsEditor
from processing.edge_detection import EdgeDetector
from processing.color_detection import ColorDetector
from processing.enhancement import ImageEnhancer
from processing.video_panorama import VideoPanorama
from utils.measurement import Measurement
from utils.file_operations import FileOperations
from utils.pdf_report import PDFReportGenerator

# Import new utilities
from utils import (
    log_info, log_error, log_warning, log_debug,
    error_handler, safe_execute, config_manager,
    performance_timer, memory_cleanup
) # importing the reuired modules and the classes, functions from the files

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    # Use getattr to avoid linter error for _MEIPASS
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

class EdgeDetectionApp:
    """
    Enhanced main application class for the Edge Detection Tool.
    Handles UI setup, event bindings, and core logic for image and video processing.
    """
    def __init__(self, root):# the constructor to initiate and call the functions and methods which are required to be executed at the beginning of the script
        """
        Initialize the main application window and all components.
        """
        self.root = root
        self.root.title("Edge Detection Tool v4.0") # the title to be shown in the window
        
        # Initialize application state
        self._initialize_app_state() 
        # Set window icon with improved resource handling
        self._set_window_icon()
        
        # Load configuration and set window size
        self._load_configuration()
        
        log_info("Starting Edge Detection Tool", version="3.0")
        
        # Initialize variables and components
        self.initialize_variables()
        self._initialize_components()
        self._setup_ui()
        self._setup_bindings()
        self._setup_performance_monitoring()
        
        # Apply saved configuration
        self._apply_saved_configuration()
        
        log_info("Application initialized successfully")
    
    def _initialize_app_state(self): # this initializes variables as empty lists, dictionaries, and is_closing as False
        """Initialize application state variables."""
        self.is_closing = False
        self.performance_stats = {}
        self.ui_update_queue = []
        self.background_tasks = []
    
    def _load_configuration(self):
        """Load application configuration."""
        try:
            ui_config = config_manager.get_ui_config()
            self.root.geometry(f"{ui_config.window_width}x{ui_config.window_height}")
            
            # Apply theme if specified
            if hasattr(ui_config, 'theme') and ui_config.theme != 'default':
                try:
                    self.root.tk.call('source', f'themes/{ui_config.theme}.tcl')
                except:
                    log_warning(f"Could not load theme: {ui_config.theme}")
                    
        except Exception as e:
            log_error("Failed to load configuration", exception=e)
            # Use default window size
            self.root.geometry("1200x800")
    
    def _set_window_icon(self):
        """Set the window icon with proper resource path handling"""
        icon_loaded = False
        
        # List of possible icon locations
        icon_paths = [
            'app_icon.ico',
            'app_icon.png',
            'icon/app_icon.ico',
            'icon/app_icon.png',
            'icon/icon.ico',
            'icon/icon.png'
        ]
        
        for icon_path in icon_paths:
            try:
                # Get the proper resource path
                full_path = get_resource_path(icon_path)
                
                if os.path.exists(full_path):
                    if icon_path.endswith('.ico'):
                        self.root.iconbitmap(full_path)
                        log_debug(f"Successfully loaded .ico icon from: {full_path}")
                        icon_loaded = True
                        break
                    else:
                        # For PNG files, use iconphoto
                        try:
                            from PIL import Image, ImageTk
                            icon_image = Image.open(full_path)
                            # Resize to standard icon size if needed
                            icon_image = icon_image.resize((32, 32), Image.Resampling.LANCZOS)
                            icon_photo = ImageTk.PhotoImage(icon_image)
                            self.root.iconphoto(True, icon_photo)
                            # Keep a reference to prevent garbage collection
                            self.icon_photo = icon_photo
                            log_debug(f"Successfully loaded .png icon from: {full_path}")
                            icon_loaded = True
                            break
                        except ImportError:
                            log_warning("PIL not available for PNG icon loading")
                            continue
                else:
                    log_debug(f"Icon not found at: {full_path}")
            except Exception as e:
                log_warning(f"Failed to load icon from {icon_path}: {e}")
                continue
        
        if not icon_loaded:
            log_warning("No application icon could be loaded")
            # Try to create a simple default icon
            try:
                self._create_default_icon()
            except Exception as e:
                log_warning(f"Could not create default icon: {e}")
    
    def _create_default_icon(self):
        """Create a simple default icon if no icon file is found"""
        try:
            from PIL import Image, ImageDraw, ImageTk
            
            # Create a simple 32x32 icon
            icon_image = Image.new('RGBA', (32, 32), (70, 130, 180, 255))  # Steel blue
            draw = ImageDraw.Draw(icon_image)
            
            # Draw a simple "E" for Edge Detection
            draw.rectangle([8, 6, 24, 10], fill='white')  # Top line
            draw.rectangle([8, 6, 12, 26], fill='white')  # Vertical line
            draw.rectangle([8, 14, 20, 18], fill='white') # Middle line
            draw.rectangle([8, 22, 24, 26], fill='white') # Bottom line
            
            icon_photo = ImageTk.PhotoImage(icon_image)
            self.root.iconphoto(True, icon_photo)
            self.icon_photo = icon_photo  # Keep reference
            log_debug("Created default icon")
        except Exception as e:
            log_warning(f"Could not create default icon: {e}")
    
    def initialize_variables(self):
        """
        Initialize all application variables and state with improved organization.
        """
        # Basic image variables
        self.img = None
        self.original_img = None
        self.edge_img = None
        self.original_image = None
        self.processed_image = None
        self.enhanced_image = None
        
        # Load edge detection configuration
        edge_config = config_manager.get_edge_detection_config()
        
        # Edge detection parameters with config defaults
        self.contrast = tk.DoubleVar(value=1.0)
        self.brightness = tk.IntVar(value=0)
        self.use_enhancement = tk.BooleanVar(value=False)
        self.edge_method = tk.StringVar(value=edge_config.method)
        self.threshold1 = tk.IntVar(value=edge_config.threshold1)
        self.threshold2 = tk.IntVar(value=edge_config.threshold2)
        self.apertureSize = tk.IntVar(value=edge_config.aperture_size)
        self.invert_edges = tk.BooleanVar(value=edge_config.invert_edges)
        self.edge_thickness = tk.IntVar(value=edge_config.edge_thickness)
        
        # Load color selection configuration
        color_config = config_manager.get_color_selection_config()
        
        # Color detection parameters with config defaults
        self.color_tolerance = tk.IntVar(value=color_config.color_tolerance)
        self.selection_mode = tk.StringVar(value=color_config.selection_mode)
        
        # Load contour configuration
        contour_config = config_manager.get_contour_config()
        
        # Contour detection parameters with config defaults
        self.contour_mode = tk.StringVar(value=contour_config.contour_mode)
        self.contour_type = tk.StringVar(value=contour_config.contour_type)
        self.contour_display_mode = tk.StringVar(value=contour_config.contour_display_mode)
        self.min_contour_area = tk.IntVar(value=contour_config.min_contour_area)
        self.max_contour_area = tk.IntVar(value=contour_config.max_contour_area)
        self.area_filter_enabled = tk.BooleanVar(value=contour_config.area_filter_enabled)
        self.area_sort_mode = tk.StringVar(value=contour_config.area_sort_mode)
        self.contour_source = tk.StringVar(value=contour_config.contour_source)
        
        # Contour processing options
        self.use_convex_hull = tk.BooleanVar(value=contour_config.use_convex_hull)
        self.convex_hull_method = tk.StringVar(value=contour_config.convex_hull_method)
        self.convex_hull_tolerance = tk.DoubleVar(value=contour_config.convex_hull_tolerance)
        self.convex_hull_smoothing = tk.DoubleVar(value=contour_config.convex_hull_smoothing)
        
        # Load measurement configuration
        measure_config = config_manager.get_measurement_config()
        
        # Measurement parameters with config defaults
        self.measure_mode = tk.BooleanVar(value=False)
        self.pixel_to_mm_ratio = tk.DoubleVar(value=measure_config.pixel_to_mm_ratio)
        self.is_measuring = False
        self.measure_start_x = 0
        self.measure_start_y = 0
        self.measure_end_x = 0
        self.measure_end_y = 0
        self.measure_line_id = None
        self.measure_text_id = None
        self.measurement_px = 0
        self.measurement_mm = 0
        self.measure_target = tk.StringVar(value=measure_config.measure_target)
        self.realtime_measure_display = tk.BooleanVar(value=measure_config.realtime_display)
        
        # For backwards compatibility with original code
        self.original_img_data = None
        self.processed_img_data = None
        self.status_var = tk.StringVar(value="Ready. Open an image to begin.")
        self.blur_size = tk.IntVar(value=edge_config.blur_size)
        self.color_overlay = tk.BooleanVar(value=edge_config.color_overlay)
        self.edge_color = tk.StringVar(value=edge_config.edge_color)
        
        # Enhancement variables
        enhance_config = config_manager.get_enhancement_config()
        self.enhance_contrast = tk.BooleanVar(value=enhance_config.use_enhancement)
        self.contrast_level = tk.DoubleVar(value=enhance_config.contrast)
        self.brightness_level = tk.IntVar(value=enhance_config.brightness)
        
        # Variables for color selection
        self.masked_image = None
        
        # Color selection is now fully managed by ColorDetector.state
        # Legacy storage removed to prevent conflicts
        
        # Initialize zoom scales with improved zoom system
        self.original_scale = 1.0
        self.result_scale = 1.0
        
        # Enhanced zoom system variables
        self.zoom_min = 0.01  # 1% minimum zoom
        self.zoom_max = 10.0  # 1000% maximum zoom
        self.zoom_step_small = 1.1  # Small zoom step (10%)
        self.zoom_step_large = 1.5  # Large zoom step (50%)
        self.zoom_smooth_factor = 0.95  # Smoothing factor for animations
        self.zoom_animation_speed = 10  # Animation speed in milliseconds
        
        # Zoom history for undo/redo
        self.zoom_history = []
        self.zoom_history_index = -1
        self.max_zoom_history = 20
        
        # Zoom performance optimization
        self.zoom_cache = {}  # Cache for zoomed images
        self.max_zoom_cache_size = 10
        self.zoom_update_pending = False
        self.zoom_update_timer = None
        
        # Performance tracking
        self.last_update_time = 0
        self.update_frequency = 60  # Max updates per second
        
        # Color detection performance tracking
        self.color_detection_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_operations': 0
        }
        
        # Contour information tracking
        self.contour_info = None
        self.contour_visibility = []
    
    def _initialize_components(self):
        """
        Initialize processing and utility components with error handling.
        """
        try:
            self.file_ops = FileOperations(self)
            self.edge_detector = EdgeDetector(self)
            self.image_enhancer = ImageEnhancer(self)
            self.color_detector = ColorDetector(self)
            self.measurement = Measurement(self)
            self.video_panorama = VideoPanorama(self)
            self.pdf_report_generator = PDFReportGenerator(self)
            self.video_dialog = None  # Will be initialized when needed
            
            log_debug("All components initialized successfully")
        except Exception as e:
            log_error("Failed to initialize components", exception=e)
            raise
    
    def _setup_ui(self):
        """
        Set up the main UI components with improved error handling.
        """
        try:
            # Create main frame
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create the UI components in the correct order for proper visibility
            # Toolbar should be at the top
            self.toolbar = Toolbar(self)
            
            # Then the image display which takes up most of the space
            self.image_display = ImageDisplay(self)
            
            # Then the control panels
            self.control_panels = ControlPanels(self)
            
            # Create status bar at the very bottom
            self._create_status_bar()
            
            log_debug("UI setup completed successfully")
        except Exception as e:
            log_error("Failed to setup UI", exception=e)
            raise
    
    def _create_status_bar(self):
        """Create enhanced status bar with multiple sections."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main status label
        self.status_bar = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Performance info
        self.performance_label = ttk.Label(
            status_frame, 
            text="", 
            relief=tk.SUNKEN, 
            anchor=tk.E,
            width=20
        )
        self.performance_label.pack(side=tk.RIGHT, padx=2)
        
        # Zoom info bar
        self.zoom_info = ttk.Label(
            status_frame, 
            text="Zoom: 100%", 
            relief=tk.SUNKEN, 
            anchor=tk.E,
            width=15
        )
        self.zoom_info.pack(side=tk.RIGHT, padx=2)
    
    def _setup_bindings(self):
        """
        Set up all key and window event bindings with improved error handling.
        """
        try:
            # Bind resize event to update images
            self.root.bind("<Configure>", self.on_window_resize)
            
            # Add key bindings for refresh
            self.root.bind("<F5>", lambda event: self.force_refresh())
            self.root.bind("<Control-r>", lambda event: self.force_refresh())
            
            # Add keyboard zoom controls
            self.root.bind("<Control-plus>", lambda e: self.zoom_in_active())
            self.root.bind("<Control-minus>", lambda e: self.zoom_out_active())
            self.root.bind("<Control-0>", lambda e: self.reset_zoom_active())
            
            # Additional zoom keyboard shortcuts
            self.root.bind("<Control-equal>", lambda e: self.zoom_in_active())  # For keyboards without plus
            self.root.bind("<Control-Shift-plus>", lambda e: self.zoom_both_in())  # Zoom both in
            self.root.bind("<Control-Shift-equal>", lambda e: self.zoom_both_in())  # Alternative
            self.root.bind("<Control-Shift-minus>", lambda e: self.zoom_both_out())  # Zoom both out
            self.root.bind("<Control-Shift-0>", lambda e: self.reset_both_zoom())  # Reset both
            
            # Enhanced zoom controls
            self.root.bind("<Control-Up>", lambda e: self.zoom_in_active())  # Alternative zoom in
            self.root.bind("<Control-Down>", lambda e: self.zoom_out_active())  # Alternative zoom out
            self.root.bind("<Control-Home>", lambda e: self.fit_to_window())  # Fit to window
            self.root.bind("<Control-End>", lambda e: self.fit_to_width())  # Fit to width
            
            # Zoom history controls
            self.root.bind("<Control-z>", lambda e: self.zoom_undo())  # Undo zoom
            self.root.bind("<Control-y>", lambda e: self.zoom_redo())  # Redo zoom
            
            # Quick zoom percentage shortcuts
            self.root.bind("<Control-1>", lambda e: self.zoom_both_to_percentage(25))   # 25%
            self.root.bind("<Control-2>", lambda e: self.zoom_both_to_percentage(50))   # 50%
            self.root.bind("<Control-3>", lambda e: self.zoom_both_to_percentage(75))   # 75%
            self.root.bind("<Control-4>", lambda e: self.zoom_both_to_percentage(100))  # 100%
            self.root.bind("<Control-5>", lambda e: self.zoom_both_to_percentage(150))  # 150%
            self.root.bind("<Control-6>", lambda e: self.zoom_both_to_percentage(200))  # 200%
            self.root.bind("<Control-7>", lambda e: self.zoom_both_to_percentage(300))  # 300%
            self.root.bind("<Control-8>", lambda e: self.zoom_both_to_percentage(400))  # 400%
            self.root.bind("<Control-9>", lambda e: self.zoom_both_to_percentage(500))  # 500%
            
            # Viewport control shortcuts
            self.root.bind("<Control-f>", lambda e: self.smart_view())  # Smart view
            
            # Panels window management
            self.root.bind("<Control-Alt-r>", lambda e: self.reset_panels_window_position())  # Reset panels position
            
            # Debug controls
            self.root.bind("<Control-Alt-e>", lambda e: self.enable_all_toolbar_buttons())  # Enable all toolbar buttons
            
            # Application lifecycle
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Focus events for better user experience
            self.root.bind("<FocusIn>", self._on_focus_in)
            self.root.bind("<FocusOut>", self._on_focus_out)
            
            log_debug("Event bindings setup completed")
        except Exception as e:
            log_error("Failed to setup bindings", exception=e)
    
    def _setup_performance_monitoring(self):
        """Set up performance monitoring and optimization."""
        self._schedule_performance_update()
        self._schedule_memory_cleanup()
    
    def _schedule_performance_update(self):
        """Schedule periodic performance updates."""
        if not self.is_closing:
            self._update_performance_display()
            self.root.after(2000, self._schedule_performance_update)  # Update every 2 seconds
    
    def _schedule_memory_cleanup(self):
        """Schedule periodic memory cleanup."""
        if not self.is_closing:
            memory_cleanup()
            self.root.after(30000, self._schedule_memory_cleanup)  # Cleanup every 30 seconds
    
    def _update_performance_display(self):
        """Update performance information in status bar with measurement stats."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            # Update color detection stats if available
            color_stats = ""
            if hasattr(self, 'color_detector'):
                cache_stats = self.color_detector.get_cache_stats()
                if cache_stats.get('cache_size', 0) > 0:
                    hit_rate = cache_stats.get('hit_rate', 0)
                    color_stats = f" | Color Cache: {hit_rate}%"
            
            # Add measurement performance stats if available
            measurement_stats = ""
            if hasattr(self, 'measurement'):
                try:
                    cache_stats = self.measurement.get_cache_stats()
                    if cache_stats.get('coord_cache_hit_rate', 0) > 0:
                        measurement_stats = f" | Measure Cache: {cache_stats['coord_cache_hit_rate']}%"
                except Exception as e:
                    log_warning("Failed to get measurement cache stats", exception=e)
            
            perf_text = f"CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.1f}MB{color_stats}{measurement_stats}"
            self.performance_label.config(text=perf_text)
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            log_warning("Failed to update performance display", exception=e)
    
    def _apply_saved_configuration(self):
        """Apply basic saved configuration to UI elements."""
        try:
            log_info("Starting to apply saved configuration")
            
            # Sync enhancement settings
            self.contrast.set(self.contrast_level.get())
            self.brightness.set(self.brightness_level.get())
            self.use_enhancement.set(self.enhance_contrast.get())
            log_debug(f"Enhancement settings synced: contrast={self.contrast.get()}, brightness={self.brightness.get()}")

            # Restore selected colors if they exist (UI only, no processing)
            self._restore_selected_colors()

            # Sync contour checkbox with configuration
            if hasattr(self, 'control_panels'):
                self.control_panels.sync_contour_checkbox()

            log_info("Applied saved configuration successfully")
        except Exception as e:
            log_error("Failed to apply saved configuration", exception=e)
    
    def _apply_saved_configuration_after_image_load(self):
        """Apply saved configuration that requires an image to be loaded."""
        try:
            log_info("Starting to apply saved configuration after image load")
            
            # Apply edge detection if an image is loaded to show the result immediately
            if self.original_image is not None and hasattr(self, 'edge_detector'):
                log_info(f"Original image exists, applying processing")
                
                # Apply color mask if colors exist
                if hasattr(self, 'color_detector'):
                    visible_colors, _ = self.color_detector.state.get_visible_colors()
                    if visible_colors:
                        log_info(f"Applying color mask for {len(visible_colors)} visible colors")
                        self.color_detector.apply_all_colors()
                        log_info("Color mask applied successfully")
                    else:
                        log_warning("No visible colors found")
                else:
                    log_info("No color detector available")
                
                # Always apply edge detection after color mask
                log_info("Applying edge detection")
                self.edge_detector.apply_edge_detection()
                log_info("Edge detection applied")
                
                # Ensure result image is displayed
                if self.processed_image is not None:
                    log_info("Displaying processed image")
                    self.image_display.display_image(self.processed_image, self.image_display.result_display)
                    log_info("Processed image displayed successfully")
                else:
                    log_warning("Processed image is None after edge detection")
            else:
                log_warning("No original image or edge detector available")

            log_info("Applied saved configuration after image load successfully")
        except Exception as e:
            log_error("Failed to apply saved configuration after image load", exception=e)
    
    def _restore_selected_colors(self):
        """Restore selected colors from configuration using new state synchronization"""
        try:
            log_info("Starting to restore selected colors")
            
            if not hasattr(self, 'color_detector') or not self.color_detector:
                log_warning("No color detector available for restoring colors")
                return
                
            # Get saved color configuration
            color_config = config_manager.get_color_selection_config()
            
            # Use the new state synchronization method
            self.color_detector.state.sync_with_config(color_config)
            
            # Validate and fix any state inconsistencies
            if not self.color_detector.state.validate_state():
                log_warning("Color state inconsistencies detected, fixing...")
                self.color_detector.state.fix_state_inconsistencies()
                
            log_info("Color state synchronized with configuration")
            
            # Update UI if control panels exist
            if hasattr(self, 'control_panels'):
                self.control_panels.update_color_list_display()
                log_debug("Updated color list display in control panels")
                
        except Exception as e:
            log_error("Failed to restore selected colors", exception=e)
    
    def _on_focus_in(self, event=None):
        """Handle application focus in."""
        if event and event.widget == self.root:
            log_debug("Application gained focus")
    
    def _on_focus_out(self, event=None):
        """Handle application focus out."""
        if event and event.widget == self.root:
            log_debug("Application lost focus")
    
    def reset_panels_window_position(self):
        """Reset the control panels window position to avoid overlap."""
        try:
            if hasattr(self, 'control_panels') and self.control_panels.is_visible:
                # Note: _reset_window_position method not available in current ControlPanels implementation
                self.status_var.set("Control panels window position reset")
                log_debug("Control panels window position reset from main application")
        except Exception as e:
            log_error("Failed to reset panels window position", exception=e)
    
    # --- UI and Event Handlers ---
    def on_window_resize(self, event):
        """
        Handle window resize events to update image display with improved performance.
        """
        # Only process if it's a window resize, not a widget resize
        if event.widget == self.root:
            # Implement resize delay to avoid excessive recalculations
            if hasattr(self, 'resize_timer') and self.resize_timer:
                self.root.after_cancel(self.resize_timer)
            
            # Set a short timer to only update after resize has stopped for a moment
            self.resize_timer = self.root.after(200, self._apply_after_resize)
    
    def _apply_after_resize(self):
        """
        Apply changes after window resize with a delay to avoid excessive recalculations.
        """
        try:
            # Clear the timer reference
            self.resize_timer = None
            # Update images with new window dimensions
            if self.original_image is not None:
                # Redisplay images to adjust to new window size
                self.edge_detector.apply_edge_detection()
            
            # Update panels window position if it exists and is visible
            if hasattr(self, 'control_panels') and self.control_panels.is_visible:
                # Small delay to ensure main window has finished resizing
                # Note: _reset_window_position method not available in current ControlPanels implementation
                pass
                
            log_debug("Window resize applied")
        except Exception as e:
            log_error("Failed to apply window resize", exception=e)
    
    # --- Settings and Refresh ---
    @error_handler(error_message="Failed to reset settings")
    @performance_timer("reset_settings")
    def reset_settings(self):
        """
        Reset all settings to their default values and update the UI accordingly.
        Optimized for performance by batching operations and deferring expensive processing.
        """
        # Get all configs at once to reduce I/O overhead
        configs = config_manager.get_all_configs()
        edge_config = configs['edge_detection']
        enhance_config = configs['enhancement']
        measure_config = configs['measurement']
        color_config = configs['color_selection']
        
        # Disable UI updates temporarily to batch them
        ui_update_pending = False
        
        try:
            # Reset edge detection settings (batch all at once)
            self.threshold1.set(edge_config.threshold1)
            self.threshold2.set(edge_config.threshold2)
            self.blur_size.set(edge_config.blur_size)
            self.edge_thickness.set(edge_config.edge_thickness)
            self.color_overlay.set(edge_config.color_overlay)
            self.edge_color.set(edge_config.edge_color)
            self.edge_method.set(edge_config.method)
            self.invert_edges.set(edge_config.invert_edges)
            
            # Reset enhancement settings (batch all at once)
            self.enhance_contrast.set(enhance_config.use_enhancement)
            self.use_enhancement.set(enhance_config.use_enhancement)
            self.contrast.set(enhance_config.contrast)
            self.brightness.set(enhance_config.brightness)
            self.contrast_level.set(enhance_config.contrast)
            self.brightness_level.set(enhance_config.brightness)
            
            # Reset measurement settings (batch all at once)
            self.measure_mode.set(False)
            self.pixel_to_mm_ratio.set(measure_config.pixel_to_mm_ratio)
            self.measure_target.set(measure_config.measure_target)
            
            # Reset color selection settings
            self.color_tolerance.set(color_config.color_tolerance)
            
            # Reset contour settings
            contour_config = config_manager.get_contour_config()
            self.contour_mode.set(contour_config.contour_mode)
            self.contour_type.set(contour_config.contour_type)
            self.contour_display_mode.set(contour_config.contour_display_mode)
            self.min_contour_area.set(contour_config.min_contour_area)
            self.max_contour_area.set(contour_config.max_contour_area)
            self.area_filter_enabled.set(contour_config.area_filter_enabled)
            self.area_sort_mode.set(contour_config.area_sort_mode)
            self.contour_source.set(contour_config.contour_source)
            self.use_convex_hull.set(contour_config.use_convex_hull)
            

            
            # Reset zoom levels to fit width to viewport
            if self.original_image is not None:
                # Defer zoom reset to the deferred processing to avoid blocking UI
                pass  # Will be handled in _deferred_reset_processing
            else:
                # If no image, just reset to 1.0
                self.original_scale = 1.0
                self.result_scale = 1.0
            
            # Clear enhanced image if it exists (memory cleanup)
            if hasattr(self, 'enhanced_image'):
                delattr(self, 'enhanced_image')
            
            # Reset color selection efficiently
            if hasattr(self, 'color_detector'):
                # Use fast reset to avoid expensive operations
                self.color_detector.fast_reset_selection()
                
                # Clear saved colors in configuration (batch update)
                config_manager.update_config('color_selection',
                    saved_selected_colors=[],
                    saved_color_names=[],
                    saved_color_visible=[],
                    use_hsv_color_space=False
                )
                
                # Clear contour configuration
                config_manager.update_config('contour',
                    contour_mode="none",
                    contour_type="external",
                    contour_display_mode="over_original",
                    min_contour_area=500,
                    max_contour_area=0,
                    area_filter_enabled=True,
                    area_sort_mode="none",
                    contour_source="edges",
                    use_convex_hull=False,
                    convex_hull_method="standard",
                    convex_hull_tolerance=0.02,
                    convex_hull_smoothing=0.0
                )
            
            # Clear measurements efficiently
            if hasattr(self, 'measurement'):
                self.measurement.fast_clear_measurements()
            
            # Update status immediately
            self.status_var.set("Settings reset to default values")
            
            # Update zoom info (fast operation)
            self.update_zoom_info()
            
            # Defer expensive UI updates and processing to avoid blocking UI
            if self.original_image is not None:
                # Schedule edge detection and display updates for next event loop
                self.root.after(10, self._deferred_reset_processing)
            else:
                # If no image, still do UI updates but defer them
                self.root.after(10, self._deferred_ui_updates)
                log_info("Settings reset to defaults (no image loaded)")
                
        except Exception as e:
            log_error("Error during settings reset", exception=e)
            self.status_var.set("Error resetting settings")
        finally:
            # Ensure UI updates are processed
            if ui_update_pending:
                self.root.update_idletasks()
    
    def _deferred_ui_updates(self):
        """
        Deferred UI updates after settings reset to avoid blocking the UI.
        This method runs in the next event loop cycle.
        """
        try:
            # Update measurement UI
            if hasattr(self, 'control_panels'):
                self.control_panels.update_measurement_ui()
            
            # Sync contour checkbox
            if hasattr(self, 'control_panels'):
                self.control_panels.sync_contour_checkbox()
            
            # Update color list display
            if hasattr(self, 'color_detector'):
                self.color_detector.update_color_list_display()
            
            log_info("Deferred UI updates completed")
            
        except Exception as e:
            log_error("Error during deferred UI updates", exception=e)
    
    def _deferred_reset_processing(self):
        """
        Deferred processing after settings reset to avoid blocking the UI.
        This method runs in the next event loop cycle.
        """
        try:
            # Clear any masked image to ensure clean reset
            if hasattr(self, 'masked_image'):
                delattr(self, 'masked_image')
            
            # Clear image display cache to ensure fresh display
            if hasattr(self, 'image_display') and hasattr(self.image_display, 'image_cache'):
                self.image_display.image_cache.clear()
            
            # Display original image without enhancements
            if hasattr(self, 'image_display') and self.original_image is not None:
                self.image_display.display_image(self.original_image, self.image_display.original_display)
            
            # Reset zoom to fit width to viewport
            if self.original_image is not None:
                self._fit_width_to_canvas('original')
                self._fit_width_to_canvas('result')
            
            # Apply edge detection with reset settings using fast mode for better performance
            if hasattr(self, 'edge_detector'):
                self.edge_detector.apply_edge_detection(fast_mode=True)
            
            # Now do UI updates after processing is complete
            self._deferred_ui_updates()
            
            log_info("Settings reset to defaults and processing completed")
            
        except Exception as e:
            log_error("Error in deferred reset processing", exception=e)
            self.status_var.set("Error completing settings reset")
    
    @error_handler(error_message="Failed to refresh display")
    def force_refresh(self):
        """
        Force a refresh of the display, clearing any caches and redisplaying images.
        """
        # Redisplay images if available
        if self.original_image is not None:
            self.image_display.display_original_image()
            self.edge_detector.apply_edge_detection()
        self.status_var.set("Display refreshed")
        log_debug("Display refreshed")
    
    def recover_from_ui_stuck(self):
        """
        Recover from UI stuck state by resetting update flags and clearing caches.
        """
        try:
            log_warning("Attempting UI stuck recovery")
            
            # Reset update flags in edge detector
            if hasattr(self, 'edge_detector'):
                if hasattr(self.edge_detector, '_ui_update_in_progress'):
                    self.edge_detector._ui_update_in_progress = False
                if hasattr(self.edge_detector, '_last_ui_update_time'):
                    delattr(self.edge_detector, '_last_ui_update_time')
            
            # Reset update flags in control panels
            if hasattr(self, 'control_panels'):
                if hasattr(self.control_panels, '_contour_update_in_progress'):
                    self.control_panels._contour_update_in_progress = False
                if hasattr(self.control_panels, '_updating_contour_details'):
                    self.control_panels._updating_contour_details = False
            
            # Reset measurement system
            if hasattr(self, 'measurement'):
                self.measurement.recover_from_ui_stuck()
            
            # Clear any pending UI updates
            if hasattr(self, 'root') and self.root:
                try:
                    # Cancel any pending after() calls
                    self.root.after_cancel("all")
                except:
                    pass  # Ignore errors in cleanup
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Update status
            self.status_var.set("UI recovered from stuck state")
            log_info("UI stuck recovery completed")
            
        except Exception as e:
            log_error("Error in UI stuck recovery", exception=e)
            self.status_var.set("UI recovery failed")
    
    def optimize_contour_display_for_large_counts(self):
        """
        Automatically optimize contour display settings when large numbers of contours are detected.
        """
        try:
            if hasattr(self, 'contour_info') and self.contour_info:
                contour_count = self.contour_info.get('filtered_contours', 0)
                
                if contour_count > 500:
                    log_info(f"Large contour count detected ({contour_count}), suggesting optimizations")
                    
                    # Suggest area filtering if not already enabled
                    if hasattr(self, 'area_filter_enabled') and not self.area_filter_enabled.get():
                        self.status_var.set(f"Large contour count ({contour_count}) detected. Consider enabling area filtering for better performance.")
                        
                        # Show a suggestion to the user
                        if hasattr(self, 'control_panels'):
                            try:
                                # Try to enable area filtering automatically with reasonable defaults
                                if hasattr(self, 'area_filter_enabled'):
                                    self.area_filter_enabled.set(True)
                                    
                                    # Set reasonable area limits based on image size
                                    if hasattr(self, 'original_image') and self.original_image is not None:
                                        image_size = self.original_image.shape[0] * self.original_image.shape[1]
                                        if image_size > 10000000:  # 10MP+
                                            # For very large images, use more restrictive limits
                                            if hasattr(self, 'min_contour_area'):
                                                self.min_contour_area.set(50000)
                                            if hasattr(self, 'max_contour_area'):
                                                self.max_contour_area.set(500000)
                                        else:
                                            # For smaller images, use moderate limits
                                            if hasattr(self, 'min_contour_area'):
                                                self.min_contour_area.set(1000)
                                            if hasattr(self, 'max_contour_area'):
                                                self.max_contour_area.set(0)  # 0 = no limit
                                    
                                    self.status_var.set(f"Area filtering enabled automatically for {contour_count} contours")
                                    log_info("Area filtering enabled automatically for large contour count")
                            except Exception as e:
                                log_error("Error auto-enabling area filtering", exception=e)
                    
                    # Suggest reducing contour count
                    elif contour_count > 1000:
                        self.status_var.set(f"Very large contour count ({contour_count}). Consider using more restrictive area filtering.")
                
        except Exception as e:
            log_error("Error optimizing contour display", exception=e)
    
    # --- Zoom Controls ---
    def zoom_in_active(self, event=None):
        """
        Zoom in on the currently active image canvas with improved controls.
        """
        try:
            active_widget = self.root.focus_get()
            if active_widget == self.image_display.original_canvas:
                self._zoom_canvas('original', self.zoom_step_small)
            elif active_widget == self.image_display.result_canvas:
                self._zoom_canvas('result', self.zoom_step_small)
            else:
                # Default to original canvas if no focus
                self._zoom_canvas('original', self.zoom_step_small)
        except Exception as e:
            log_error("Failed to zoom in", exception=e)
    
    def zoom_out_active(self, event=None):
        """
        Zoom out on the currently active image canvas with improved controls.
        """
        try:
            active_widget = self.root.focus_get()
            if active_widget == self.image_display.original_canvas:
                self._zoom_canvas('original', 1.0 / self.zoom_step_small)
            elif active_widget == self.image_display.result_canvas:
                self._zoom_canvas('result', 1.0 / self.zoom_step_small)
            else:
                # Default to original canvas if no focus
                self._zoom_canvas('original', 1.0 / self.zoom_step_small)
        except Exception as e:
            log_error("Failed to zoom out", exception=e)
    
    def reset_zoom_active(self, event=None):
        """
        Reset zoom to fit image width to the viewport on the currently active image canvas.
        """
        try:
            active_widget = self.root.focus_get()
            if active_widget == self.image_display.original_canvas:
                self._fit_width_to_canvas('original')
            elif active_widget == self.image_display.result_canvas:
                self._fit_width_to_canvas('result')
            else:
                # Default to original canvas if no focus
                self._fit_width_to_canvas('original')
        except Exception as e:
            log_error("Failed to reset zoom", exception=e)
    
    def zoom_both_in(self, event=None):
        """
        Zoom in on both original and result image canvases.
        """
        try:
            self._zoom_canvas('original', self.zoom_step_small)
            self._zoom_canvas('result', self.zoom_step_small)
        except Exception as e:
            log_error("Failed to zoom both in", exception=e)
    
    def zoom_both_out(self, event=None):
        """
        Zoom out on both original and result image canvases.
        """
        try:
            self._zoom_canvas('original', 1.0 / self.zoom_step_small)
            self._zoom_canvas('result', 1.0 / self.zoom_step_small)
        except Exception as e:
            log_error("Failed to zoom both out", exception=e)
    
    def reset_both_zoom(self, event=None):
        """
        Reset zoom to fit image width to the viewport on both original and result image canvases.
        """
        try:
            self._fit_width_to_canvas('original')
            self._fit_width_to_canvas('result')
        except Exception as e:
            log_error("Failed to reset both zoom", exception=e)
    
    def _fit_width_to_canvas(self, canvas_type):
        """
        Fit image width to the viewport for a specific canvas.
        """
        try:
            if self.original_image is None:
                return
                
            # Get canvas dimensions
            if canvas_type == 'original':
                canvas = self.image_display.original_canvas
                current_scale = self.original_scale
            else:
                canvas = self.image_display.result_canvas
                current_scale = self.result_scale
            
            if canvas is None:
                return
                
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Use default dimensions if canvas hasn't been drawn yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 400
            
            img_height, img_width = self.original_image.shape[:2]
            
            # Calculate scale to fit width
            fit_scale = canvas_width / img_width
            
            # Don't zoom in beyond 200% for width fitting
            fit_scale = min(fit_scale, 2.0)
            
            # Set a reasonable minimum scale (don't go below 5% zoom)
            fit_scale = max(fit_scale, 0.05)
            
            # Apply the scale
            if canvas_type == 'original':
                self.original_scale = fit_scale
                self._update_original_display()
            else:
                self.result_scale = fit_scale
                self._update_result_display()
            
            # Reset scroll position to top-left
            canvas.xview_moveto(0)
            canvas.yview_moveto(0)
            
            # Update zoom info
            self.update_zoom_info()
            
            # Redraw measurements if they exist
            self._redraw_measurements_after_zoom()
            
            zoom_percent = int(fit_scale * 100)
            self.status_var.set(f"Reset zoom: {canvas_type} image width fitted to viewport (Zoom: {zoom_percent}%)")
            
        except Exception as e:
            log_error(f"Failed to fit width to canvas for {canvas_type}", exception=e)
    
    def zoom_to_point(self, canvas_type, x, y, zoom_factor):
        """
        Zoom to a specific point on the canvas, keeping that point centered.
        """
        try:
            if canvas_type == 'original':
                canvas = self.image_display.original_canvas
                current_scale = self.original_scale
            else:
                canvas = self.image_display.result_canvas
                current_scale = self.result_scale
            
            # Check if canvas exists
            if canvas is None:
                log_warning(f"Cannot zoom to point: {canvas_type} canvas is None")
                return
            
            # Calculate new scale with limits
            new_scale = current_scale * zoom_factor
            new_scale = max(self.zoom_min, min(self.zoom_max, new_scale))
            
            if new_scale != current_scale:
                # Calculate the point in canvas coordinates
                canvas_x = canvas.canvasx(x)
                canvas_y = canvas.canvasy(y)
                
                # Apply zoom
                if canvas_type == 'original':
                    self._set_zoom('original', new_scale)
                else:
                    self._set_zoom('result', new_scale)
                
                # Center the point
                self._center_point_on_canvas(canvas, canvas_x, canvas_y, current_scale, new_scale)
                
        except Exception as e:
            log_error("Failed to zoom to point", exception=e)
    
    def _center_point_on_canvas(self, canvas, x, y, old_scale, new_scale):
        """Center a specific point on the canvas after zoom."""
        try:
            # Calculate the new position to keep the point centered
            scale_ratio = new_scale / old_scale
            new_x = x * scale_ratio
            new_y = y * scale_ratio
            
            # Get canvas viewport
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Calculate scroll position to center the point
            scroll_x = max(0, new_x - canvas_width / 2)
            scroll_y = max(0, new_y - canvas_height / 2)
            
            # Apply scroll
            canvas.xview_moveto(scroll_x / canvas.winfo_reqwidth())
            canvas.yview_moveto(scroll_y / canvas.winfo_reqheight())
            
        except Exception as e:
            log_warning("Failed to center point on canvas", exception=e)
    
    def _zoom_canvas(self, canvas_type, zoom_factor):
        """
        Internal method to zoom a specific canvas with limits and history.
        """
        try:
            # Get current scale
            if canvas_type == 'original':
                current_scale = self.original_scale
            else:
                current_scale = self.result_scale
            
            # Calculate new scale with limits
            new_scale = current_scale * zoom_factor
            new_scale = max(self.zoom_min, min(self.zoom_max, new_scale))
            
            if new_scale != current_scale:
                # Add to history before changing
                self._add_to_zoom_history(canvas_type, current_scale)
                
                # Apply new scale
                self._set_zoom(canvas_type, new_scale)
                
        except Exception as e:
            log_error(f"Failed to zoom {canvas_type} canvas", exception=e)
    
    def _set_zoom(self, canvas_type, new_scale):
        """
        Set zoom level for a specific canvas with performance optimization.
        """
        try:
            # Apply scale
            if canvas_type == 'original':
                self.original_scale = new_scale
                self._update_original_display()
            else:
                self.result_scale = new_scale
                self._update_result_display()
            
            # Update zoom info
            self.update_zoom_info()
            
            # Schedule measurement redraw
            self._schedule_measurement_redraw()
            
        except Exception as e:
            log_error(f"Failed to set zoom for {canvas_type}", exception=e)
    
    def smooth_zoom_to(self, canvas_type, target_scale, duration_ms=300):
        """
        Smoothly animate zoom to a target scale over a specified duration.
        """
        try:
            # Get current scale
            if canvas_type == 'original':
                current_scale = self.original_scale
            else:
                current_scale = self.result_scale
            
            if current_scale == target_scale:
                return  # No animation needed
            
            # Calculate animation parameters
            scale_diff = target_scale - current_scale
            steps = max(1, int(duration_ms / self.zoom_animation_speed))
            step_size = scale_diff / steps
            
            def animate_step(step=0):
                if step >= steps:
                    # Animation complete, set final scale
                    self._set_zoom(canvas_type, target_scale)
                    return
                
                # Calculate current scale for this step
                current_step_scale = current_scale + (step_size * step)
                current_step_scale = max(self.zoom_min, min(self.zoom_max, current_step_scale))
                
                # Apply intermediate scale
                if canvas_type == 'original':
                    self.original_scale = current_step_scale
                    self._update_original_display()
                else:
                    self.result_scale = current_step_scale
                    self._update_result_display()
                
                # Update zoom info
                self.update_zoom_info()
                
                # Schedule next step
                self.root.after(self.zoom_animation_speed, lambda: animate_step(step + 1))
            
            # Start animation
            animate_step()
            
        except Exception as e:
            log_error(f"Failed to smooth zoom {canvas_type}", exception=e)
    
    def zoom_to_percentage(self, canvas_type, percentage):
        """
        Zoom to a specific percentage (e.g., 50 for 50%).
        """
        try:
            if percentage <= 0:
                log_warning(f"Invalid zoom percentage: {percentage}")
                return
            
            # Convert percentage to scale (e.g., 50% = 0.5)
            target_scale = percentage / 100.0
            
            # Apply limits
            target_scale = max(self.zoom_min, min(self.zoom_max, target_scale))
            
            # Set zoom
            self._set_zoom(canvas_type, target_scale)
            
            # Update status
            self.status_var.set(f"Zoomed to {percentage}%")
            
        except Exception as e:
            log_error(f"Failed to zoom to percentage {percentage}", exception=e)
    
    def zoom_both_to_percentage(self, percentage):
        """
        Zoom both canvases to a specific percentage.
        """
        try:
            self.zoom_to_percentage('original', percentage)
            self.zoom_to_percentage('result', percentage)
        except Exception as e:
            log_error(f"Failed to zoom both to percentage {percentage}", exception=e)
    
    def _update_original_display(self):
        """Update original image display with zoom optimization."""
        try:
            if self.original_image is not None:
                # Use cached zoom if available
                cache_key = f"original_{self.original_scale:.3f}"
                if cache_key in self.zoom_cache:
                    cached_image = self.zoom_cache[cache_key]
                    self.image_display.display_image(cached_image, self.image_display.original_display)
                else:
                    # Regular display (will be cached by image_display if needed)
                    self.image_display.display_original_image()
                    
                    # Clean up cache if too large
                    self._cleanup_zoom_cache()
        except Exception as e:
            log_error("Failed to update original display", exception=e)
    
    def _update_result_display(self):
        """Update result image display with zoom optimization."""
        try:
            if self.processed_image is not None:
                # Use cached zoom if available
                cache_key = f"result_{self.result_scale:.3f}"
                if cache_key in self.zoom_cache:
                    cached_image = self.zoom_cache[cache_key]
                    self.image_display.display_image(cached_image, self.image_display.result_display)
                else:
                    # Regular display
                    self.image_display.display_image(self.processed_image, self.image_display.result_display)
                    
                    # Clean up cache if too large
                    self._cleanup_zoom_cache()
        except Exception as e:
            log_error("Failed to update result display", exception=e)
    
    def _add_to_zoom_history(self, canvas_type, scale):
        """Add zoom state to history for undo/redo functionality."""
        try:
            # Create history entry
            history_entry = {
                'canvas_type': canvas_type,
                'scale': scale,
                'timestamp': time.time()
            }
            
            # Remove any entries after current index (for redo)
            if self.zoom_history_index < len(self.zoom_history) - 1:
                self.zoom_history = self.zoom_history[:self.zoom_history_index + 1]
            
            # Add new entry
            self.zoom_history.append(history_entry)
            self.zoom_history_index += 1
            
            # Limit history size
            if len(self.zoom_history) > self.max_zoom_history:
                self.zoom_history.pop(0)
                self.zoom_history_index -= 1
                
        except Exception as e:
            log_warning("Failed to add to zoom history", exception=e)
    
    def zoom_undo(self, event=None):
        """Undo the last zoom operation."""
        try:
            if self.zoom_history_index > 0:
                self.zoom_history_index -= 1
                history_entry = self.zoom_history[self.zoom_history_index]
                self._set_zoom(history_entry['canvas_type'], history_entry['scale'])
                self.status_var.set(f"Zoom undone: {int(history_entry['scale'] * 100)}%")
        except Exception as e:
            log_error("Failed to undo zoom", exception=e)
    
    def zoom_redo(self, event=None):
        """Redo the last undone zoom operation."""
        try:
            if self.zoom_history_index < len(self.zoom_history) - 1:
                self.zoom_history_index += 1
                history_entry = self.zoom_history[self.zoom_history_index]
                self._set_zoom(history_entry['canvas_type'], history_entry['scale'])
                self.status_var.set(f"Zoom redone: {int(history_entry['scale'] * 100)}%")
        except Exception as e:
            log_error("Failed to redo zoom", exception=e)
    
    def _cleanup_zoom_cache(self):
        """Clean up zoom cache to prevent memory issues."""
        try:
            if len(self.zoom_cache) > self.max_zoom_cache_size:
                # Remove oldest entries
                keys_to_remove = list(self.zoom_cache.keys())[:len(self.zoom_cache) - self.max_zoom_cache_size]
                for key in keys_to_remove:
                    del self.zoom_cache[key]
        except Exception as e:
            log_warning("Failed to cleanup zoom cache", exception=e)
    
    def _schedule_measurement_redraw(self):
        """Schedule measurement redraw to avoid excessive updates."""
        try:
            if self.zoom_update_pending:
                return
            
            self.zoom_update_pending = True
            if self.zoom_update_timer:
                self.root.after_cancel(self.zoom_update_timer)
            
            self.zoom_update_timer = self.root.after(100, self._redraw_measurements_after_zoom)
        except Exception as e:
            log_warning("Failed to schedule measurement redraw", exception=e)
    
    def _redraw_measurements_after_zoom(self):
        """Helper method to redraw measurements after zoom changes."""
        try:
            self.zoom_update_pending = False
            self.zoom_update_timer = None
            
            if hasattr(self, 'measurement') and self.measurement.measurements:
                # Force redraw to ensure proper positioning
                self.measurement.redraw_all_measurements()
        except Exception as e:
            log_warning("Failed to redraw measurements after zoom", exception=e)
    
    def fit_to_window(self, event=None):
        """
        Fit the image to the window viewport, showing the entire image.
        """
        try:
            if self.original_image is None:
                return
            # Ensure canvases exist
            if (
                self.image_display is None or
                self.image_display.original_canvas is None or
                self.image_display.result_canvas is None
            ):
                return
            # Get canvas dimensions
            canvas_width = self.image_display.original_canvas.winfo_width()
            canvas_height = self.image_display.original_canvas.winfo_height()
            # Use default dimensions if canvas hasn't been drawn yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 400
            img_height, img_width = self.original_image.shape[:2]
            # Calculate scale factors to fit the image in the viewport
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            # Use the smaller scale to ensure the entire image fits
            fit_scale = min(scale_x, scale_y)
            # Set a reasonable minimum scale (don't go below 5% zoom)
            fit_scale = max(fit_scale, 0.05)
            # Apply the scale to both views
            self.original_scale = fit_scale
            self.result_scale = fit_scale
            # Reset scroll position to top-left
            self.image_display.original_canvas.xview_moveto(0)
            self.image_display.original_canvas.yview_moveto(0)
            self.image_display.result_canvas.xview_moveto(0)
            self.image_display.result_canvas.yview_moveto(0)
            # Update display
            self.update_zoom_info()
            self.image_display.display_original_image()
            if self.processed_image is not None:
                self.image_display.display_image(self.processed_image, self.image_display.result_display)
            # Redraw measurements if they exist
            self._redraw_measurements_after_zoom()
            zoom_percent = int(fit_scale * 100)
            self.status_var.set(f"Image fitted to window (Zoom: {zoom_percent}%)")
        except Exception as e:
            log_error("Failed to fit image to window", exception=e)
    
    def fit_to_width(self, event=None):
        """
        Fit the image width to the window viewport, optimized for wide images.
        """
        try:
            if self.original_image is None:
                return
            # Ensure canvases exist
            if (
                self.image_display is None or
                self.image_display.original_canvas is None or
                self.image_display.result_canvas is None
            ):
                return
            # Get canvas dimensions
            canvas_width = self.image_display.original_canvas.winfo_width()
            canvas_height = self.image_display.original_canvas.winfo_height()
            # Use default dimensions if canvas hasn't been drawn yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 400
            img_height, img_width = self.original_image.shape[:2]
            # Calculate scale to fit width
            fit_scale = canvas_width / img_width
            # Don't zoom in beyond 200% for width fitting
            fit_scale = min(fit_scale, 2.0)
            # Set a reasonable minimum scale (don't go below 5% zoom)
            fit_scale = max(fit_scale, 0.05)
            # Apply the scale to both views
            self.original_scale = fit_scale
            self.result_scale = fit_scale
            # Reset scroll position to top-left
            self.image_display.original_canvas.xview_moveto(0)
            self.image_display.original_canvas.yview_moveto(0)
            self.image_display.result_canvas.xview_moveto(0)
            self.image_display.result_canvas.yview_moveto(0)
            # Update display
            self.update_zoom_info()
            self.image_display.display_original_image()
            if self.processed_image is not None:
                self.image_display.display_image(self.processed_image, self.image_display.result_display)
            # Redraw measurements if they exist
            self._redraw_measurements_after_zoom()
            zoom_percent = int(fit_scale * 100)
            self.status_var.set(f"Image width fitted to window (Zoom: {zoom_percent}%)")
        except Exception as e:
            log_error("Failed to fit image width to window", exception=e)
    
    def smart_view(self):
        """
        Intelligently fit the image to the viewport based on image aspect ratio.
        This is a unified view method that replaces the separate fit options.
        """
        try:
            if self.original_image is None:
                return
                
            img_height, img_width = self.original_image.shape[:2]
            aspect_ratio = img_width / img_height
            
            # For very wide images (aspect ratio > 2), use fit to width
            if aspect_ratio > 2.0:
                self.fit_to_width()
                self.status_var.set("Wide image - fitted to width for optimal viewing")
            # For very tall images (aspect ratio < 0.5), use fit to window
            elif aspect_ratio < 0.5:
                self.fit_to_window()  # Fit to window works well for tall images
                self.status_var.set("Tall image - fitted to window for optimal viewing")
            # For normal aspect ratios, use fit to window
            else:
                self.fit_to_window()
                self.status_var.set("Image fitted to window for optimal viewing")
                
            # Redraw measurements if they exist after the fit operation
            self._redraw_measurements_after_zoom()
                
        except Exception as e:
            log_error("Failed to smart view image", exception=e)
    
    def smart_fit_for_measurement(self):
        """
        Intelligently fit the image for measurement based on image aspect ratio.
        """
        try:
            if self.original_image is None:
                return
                
            img_height, img_width = self.original_image.shape[:2]
            aspect_ratio = img_width / img_height
            
            # For very wide images (aspect ratio > 2), use fit to width
            if aspect_ratio > 2.0:
                self.fit_to_width()
                self.status_var.set("Wide image - fitted to width for optimal measurement")
            # For very tall images (aspect ratio < 0.5), use fit to window
            elif aspect_ratio < 0.5:
                self.fit_to_window()  # Fit to window works well for tall images
                self.status_var.set("Tall image - fitted to window for optimal measurement")
            # For normal aspect ratios, use fit to window
            else:
                self.fit_to_window()
                self.status_var.set("Image fitted to window for optimal measurement")
                
            # Redraw measurements if they exist after the fit operation
            self._redraw_measurements_after_zoom()
                
        except Exception as e:
            log_error("Failed to smart fit image for measurement", exception=e)
    
    def update_zoom_info(self):
        """
        Update the zoom info label based on the current zoom levels and focus.
        """
        try:
            active_widget = self.root.focus_get()
            if active_widget == self.image_display.original_canvas:
                zoom_pct = int(self.original_scale * 100)
                self.zoom_info.config(text=f"Original: {zoom_pct}%")
            elif active_widget == self.image_display.result_canvas:
                zoom_pct = int(self.result_scale * 100)
                self.zoom_info.config(text=f"Result: {zoom_pct}%")
            else:
                orig_pct = int(self.original_scale * 100)
                res_pct = int(self.result_scale * 100)
                
                # Show sync status if zoom levels are different
                if abs(self.original_scale - self.result_scale) < 0.001:
                    self.zoom_info.config(text=f"Zoom: {orig_pct}%")
                else:
                    self.zoom_info.config(text=f"Zoom: {orig_pct}%/{res_pct}%")
                
                # Add history info if available
                if hasattr(self, 'zoom_history') and self.zoom_history:
                    history_count = len(self.zoom_history)
                    if history_count > 1:
                        self.zoom_info.config(text=f"{self.zoom_info.cget('text')} (History: {history_count})")
                        
        except Exception as e:
            log_warning("Failed to update zoom info", exception=e)
    
    # --- Dialogs ---
    @error_handler(error_message="Failed to open video panorama dialog")
    def open_video_panorama(self):
        """
        Open the video panorama dialog as independent panel window.
        """
        # Check if dialog is already open
        if hasattr(self, 'video_dialog') and self.video_dialog and self.video_dialog.is_active:
            # Bring existing window to front
            self.video_dialog.bring_to_front()
            return
        
        # Create new dialog
        self.video_dialog = VideoPanoramaDialog(self)
        self.video_dialog.show_dialog()



    @error_handler(error_message="Failed to open settings editor")
    def open_settings_editor(self):
        """Open the settings editor dialog."""
        try:
            if not hasattr(self, 'settings_editor') or self.settings_editor is None:
                self.settings_editor = SettingsEditor(self)
            self.settings_editor.show_dialog()
        except Exception as e:
            log_error("Failed to open settings editor", exception=e)
            messagebox.showerror("Error", f"Failed to open settings editor: {str(e)}")

    @error_handler(error_message="Failed to generate PDF report")
    def generate_pdf_report(self):
        """Generate a comprehensive PDF report."""
        try:
            if self.original_image is None:
                messagebox.showwarning("No Image", "Please load an image first before generating a PDF report.")
                return
            
            # Ask user for output path
            output_path = filedialog.asksaveasfilename(
                title="Save PDF Report",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                initialfile=f"edge_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            
            if not output_path:
                return  # User cancelled
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Generating PDF Report")
            progress_dialog.geometry("400x150")
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()
            
            # Set window icon
            progress_icon_image = None  # Keep reference to prevent garbage collection
            try:
                # Try to use the app icon if available
                if hasattr(self, 'icon_photo') and self.icon_photo:
                    progress_dialog.iconphoto(True, self.icon_photo)  # type: ignore[arg-type]
                else:
                    # Try to load icon from icon directory
                    icon_path = os.path.join(os.path.dirname(__file__), 'icon', 'icon-128x128.png')
                    if os.path.exists(icon_path):
                        try:
                            from PIL import Image, ImageTk
                            icon_image = Image.open(icon_path)
                            # Resize to standard icon size if needed
                            icon_image = icon_image.resize((32, 32), Image.Resampling.LANCZOS)
                            progress_icon_image = ImageTk.PhotoImage(icon_image)
                            progress_dialog.iconphoto(True, progress_icon_image)  # type: ignore[arg-type]
                        except ImportError:
                            # Fallback to tk.PhotoImage if PIL is not available
                            progress_icon_image = tk.PhotoImage(file=icon_path)
                            progress_dialog.iconphoto(True, progress_icon_image)  # type: ignore[arg-type]
            except Exception as e:
                print(f"Warning: Could not set progress dialog icon: {e}")
            
            # Center the dialog
            progress_dialog.geometry("+%d+%d" % (
                self.root.winfo_rootx() + self.root.winfo_width()//2 - 200,
                self.root.winfo_rooty() + self.root.winfo_height()//2 - 75
            ))
            
            # Progress label
            progress_label = ttk.Label(progress_dialog, text="Generating comprehensive PDF report...", font=("Arial", 10))
            progress_label.pack(pady=20)
            
            # Progress bar
            progress_bar = ttk.Progressbar(progress_dialog, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill=tk.X)
            progress_bar.start()
            
            # Status label
            status_label = ttk.Label(progress_dialog, text="Please wait...", font=("Arial", 9))
            status_label.pack(pady=10)
            
            def generate_report():
                try:
                    # Generate the PDF report
                    success = self.pdf_report_generator.generate_report(output_path, include_images=True)
                    
                    # Update UI in main thread
                    self.root.after(0, lambda: finish_generation(success))
                    
                except Exception as e:
                    log_error("Failed to generate PDF report", exception=e)
                    self.root.after(0, lambda: finish_generation(False, str(e)))
            
            def finish_generation(success, error_msg=None):
                progress_dialog.destroy()
                
                if success:
                    messagebox.showinfo("Success", f"PDF report generated successfully!\n\nSaved to: {output_path}")
                    
                    # Ask if user wants to open the PDF
                    if messagebox.askyesno("Open PDF", "Would you like to open the generated PDF report?"):
                        try:
                            import subprocess
                            import platform
                            
                            if platform.system() == "Windows":
                                os.startfile(output_path)
                            elif platform.system() == "Darwin":  # macOS
                                subprocess.run(["open", output_path])
                            else:  # Linux
                                subprocess.run(["xdg-open", output_path])
                        except Exception as e:
                            log_warning("Failed to open PDF automatically", exception=e)
                            messagebox.showinfo("Info", f"PDF saved to: {output_path}")
                else:
                    error_message = error_msg or "Failed to generate PDF report"
                    messagebox.showerror("Error", f"Failed to generate PDF report:\n\n{error_message}")
            
            # Start generation in background thread
            import threading
            thread = threading.Thread(target=generate_report, daemon=True)
            thread.start()
            
        except Exception as e:
            log_error("Failed to generate PDF report", exception=e)
            messagebox.showerror("Error", f"Failed to generate PDF report: {str(e)}")

    @error_handler(error_message="Failed to load image. Please check the file format and try again.")
    @performance_timer("image_loading")
    def load_image(self):
        """Load an image file and display it with enhanced error handling."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            log_info(f"Loading image: {file_path}")
            
            # Load and validate image
            image = cv2.imread(file_path)
            if image is None:
                log_error(f"Failed to load image: {file_path}")
                messagebox.showerror("Error", "Could not load the selected image file.")
                return
            
            # Store image data
            self.original_image = image.copy()
            self.current_image = image.copy()
            self.image_path = file_path
            
            # Update display
            self.image_display.display_original_image()
            
            # Clear color detection cache for new image
            if hasattr(self, 'color_detector'):
                self.color_detector.mask_processor.clear_cache()
                
            self.update_image_info()
            
            # Auto-fit wide images for better initial viewing
            img_height, img_width = image.shape[:2]
            aspect_ratio = img_width / img_height
            
            # If image is very wide (aspect ratio > 2.5), auto-fit to width
            if aspect_ratio > 2.5:
                self.root.after(100, lambda: self._fit_width_to_canvas('original'))  # Small delay to ensure UI is ready
                fit_info = " - Auto-fitted to width for optimal viewing"
            # If image is moderately wide (aspect ratio > 1.5), auto-fit to window
            elif aspect_ratio > 1.5:
                self.root.after(100, self.fit_to_window)  # Small delay to ensure UI is ready
                fit_info = " - Auto-fitted to window"
            else:
                # For normal aspect ratios, fit to width for consistent behavior
                self.root.after(100, lambda: self._fit_width_to_canvas('original'))
                fit_info = " - Fitted to width"
            
            # Apply saved configuration now that an image is loaded
            try:
                log_info("About to call _apply_saved_configuration_after_image_load")
                self._apply_saved_configuration_after_image_load()
                log_info("Finished calling _apply_saved_configuration_after_image_load")
            except Exception as e:
                log_error("Error in _apply_saved_configuration_after_image_load", exception=e)
            
            # Update toolbar button states after processing
            if hasattr(self, 'toolbar'):
                self.toolbar.force_state_update()
            
            # Apply fit-to-width to both original and processed images after processing
            self.root.after(200, self._apply_fit_to_width_both)
            
            # Enable controls
            if hasattr(self, 'control_panels'):
                self.control_panels.enable_controls()
            
            # Update toolbar button states
            if hasattr(self, 'toolbar'):
                self.toolbar.update_button_states()
                # Force immediate state update for file operations
                self.toolbar.force_state_update()
            
            log_info(f"Image loaded successfully: {image.shape}")
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)} ({image.shape[1]}x{image.shape[0]}){fit_info}")
            
            # Clean up memory if needed
            memory_cleanup()
    
    def update_image_info(self):
        """Update image information display."""
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]
            channels = self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1
            size_mb = (self.original_image.nbytes / 1024 / 1024)
            
            info_text = f"Size: {width}x{height}, Channels: {channels}, Memory: {size_mb:.1f}MB"
            log_debug(f"Image info: {info_text}")
    
    def save_configuration(self):
        """Save current configuration to file."""
        try:
            # Update configuration with current values
            config_manager.update_config('edge_detection',
                method=self.edge_method.get(),
                threshold1=self.threshold1.get(),
                threshold2=self.threshold2.get(),
                aperture_size=self.apertureSize.get(),
                blur_size=self.blur_size.get(),
                edge_thickness=self.edge_thickness.get(),
                invert_edges=self.invert_edges.get(),
                edge_color=self.edge_color.get(),
                color_overlay=self.color_overlay.get()
            )
            
            config_manager.update_config('enhancement',
                use_enhancement=self.use_enhancement.get(),
                contrast=self.contrast.get(),
                brightness=self.brightness.get()
            )
            
            config_manager.update_config('measurement',
                pixel_to_mm_ratio=self.pixel_to_mm_ratio.get(),
                measure_target=self.measure_target.get(),
                realtime_display=self.realtime_measure_display.get(),
                show_all_measurements=getattr(self, 'show_all_measurements', True)
            )
            
            # Save selected colors using new state methods
            color_state_data = {}
            
            if hasattr(self, 'color_detector') and self.color_detector:
                # Use the new state method to convert to config format
                color_state_data = self.color_detector.state.to_config_format()
            
            # Update color selection config with state data
            color_config_update = {
    
                'color_tolerance': self.color_tolerance.get(),
            }
            
            # Add state data if available
            if color_state_data:
                color_config_update.update(color_state_data)
                
            config_manager.update_config('color_selection', **color_config_update)
            
            # Update contour config
            config_manager.update_config('contour',
                contour_mode=self.contour_mode.get(),
                contour_type=self.contour_type.get(),
                contour_display_mode=self.contour_display_mode.get(),
                min_contour_area=self.min_contour_area.get(),
                max_contour_area=self.max_contour_area.get(),
                area_filter_enabled=self.area_filter_enabled.get(),
                area_sort_mode=self.area_sort_mode.get(),
                contour_source=self.contour_source.get(),
                use_convex_hull=self.use_convex_hull.get(),
                convex_hull_method=self.convex_hull_method.get(),
                convex_hull_tolerance=self.convex_hull_tolerance.get(),
                convex_hull_smoothing=self.convex_hull_smoothing.get(),
            )
            
            # Get current window geometry for UI config
            current_geometry = self.root.geometry()
            if current_geometry and 'x' in current_geometry:
                # Parse geometry string like "1200x800+100+200"
                parts = current_geometry.split('+')
                if len(parts) >= 2:
                    size_part = parts[0]
                    x_pos = int(parts[1]) if len(parts) > 1 else -1
                    y_pos = int(parts[2]) if len(parts) > 2 else -1
                    
                    if 'x' in size_part:
                        width, height = map(int, size_part.split('x'))
                    else:
                        width, height = 1200, 800
                else:
                    width, height = 1200, 800
                    x_pos, y_pos = -1, -1
            else:
                width, height = 1200, 800
                x_pos, y_pos = -1, -1
            
            config_manager.update_config('ui',
                window_width=width,
                window_height=height,
                panels_window_x=x_pos,
                panels_window_y=y_pos
            )
            
            # Save to file
            if config_manager.save_config():
                log_info("Configuration saved successfully")
                self.status_var.set("Configuration saved")
            else:
                log_warning("Failed to save configuration")
                self.status_var.set("Failed to save configuration")
                
        except Exception as e:
            log_error("Error saving configuration", exception=e)
            self.status_var.set("Error saving configuration")
    
    def on_closing(self):
        """Handle application closing with cleanup."""
        try:
            log_info("Application closing...")
            self.is_closing = True
            
            # Save configuration before closing
            self.save_configuration()
            
            # Cancel any pending background tasks
            for task in self.background_tasks:
                if hasattr(task, 'cancel'):
                    task.cancel()
            
            # Cleanup UI components
            if hasattr(self, 'toolbar'):
                self.toolbar.cleanup()
            
            if hasattr(self, 'control_panels'):
                self.control_panels.cleanup()
            
            # Cleanup measurement resources
            if hasattr(self, 'measurement'):
                self.measurement.cleanup_resources()
            
            # Final memory cleanup
            memory_cleanup()
            
            log_info("Application cleanup completed")
            
        except Exception as e:
            log_error("Error during application closing", exception=e)
        finally:
            self.root.destroy()

    @property
    def selected_colors(self):
        """Get selected colors from color detector state"""
        if hasattr(self, 'color_detector') and self.color_detector:
            return self.color_detector.state.selected_colors
        return []
    
    @selected_colors.setter
    def selected_colors(self, value):
        """Set selected colors in color detector state"""
        if hasattr(self, 'color_detector') and self.color_detector:
            self.color_detector.state.selected_colors = value
    
    @property
    def selected_color_names(self):
        """Get selected color names from color detector state"""
        if hasattr(self, 'color_detector') and self.color_detector:
            return self.color_detector.state.selected_color_names
        return []
    
    @selected_color_names.setter
    def selected_color_names(self, value):
        """Set selected color names in color detector state"""
        if hasattr(self, 'color_detector') and self.color_detector:
            self.color_detector.state.selected_color_names = value
    
    @property
    def current_color(self):
        """Get current color from color detector state"""
        if hasattr(self, 'color_detector') and self.color_detector:
            return self.color_detector.state.current_color
        return None
    
    @current_color.setter
    def current_color(self, value):
        """Set current color in color detector state"""
        if hasattr(self, 'color_detector') and self.color_detector:
            self.color_detector.state.current_color = value

    def _apply_fit_to_width_both(self):
        """
        Apply fit-to-width zoom to both original and processed images.
        This ensures both images have the same zoom level for optimal viewing.
        """
        try:
            if self.original_image is not None:
                # Apply fit-to-width to both canvases
                self._fit_width_to_canvas('original')
                self._fit_width_to_canvas('result')
                
                # Update status to indicate both images are fitted
                zoom_percent = int(self.original_scale * 100)
                self.status_var.set(f"Both images fitted to width (Zoom: {zoom_percent}%)")
                
                log_info("Applied fit-to-width zoom to both original and processed images")
                
        except Exception as e:
            log_error("Failed to apply fit-to-width to both images", exception=e)
    
    def enable_all_toolbar_buttons(self):
        """Debug method to enable all toolbar buttons for testing."""
        if hasattr(self, 'toolbar'):
            self.toolbar.enable_all_buttons()
            self.status_var.set("All toolbar buttons enabled for testing")
        else:
            self.status_var.set("Toolbar not available")

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop() 