# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ext
import omni.ui as ui
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade, UsdPhysics
import carb
import os
import shutil
from pathlib import Path
import asyncio
import base64
import json
import requests
from PIL import Image
import io
import time
import threading

# Import PhysX deformable utilities
try:
    from omni.physx.scripts import deformableUtils, physicsUtils
    PHYSX_AVAILABLE = True
except ImportError:
    print("PhysX deformable utilities not available")
    PHYSX_AVAILABLE = False


class ColliderType:
    """Collider type enumeration"""
    NONE = UsdPhysics.Tokens.none
    MESH_SIMPLIFICATION = UsdPhysics.Tokens.meshSimplification
    CONVEX_HULL = UsdPhysics.Tokens.convexHull
    CONVEX_DECOMPOSITION = UsdPhysics.Tokens.convexDecomposition
    BOUNDING_SPHERE = UsdPhysics.Tokens.boundingSphere
    BOUNDING_CUBE = UsdPhysics.Tokens.boundingCube


# Functions and vars are available to other extensions as usual in python:
# `hunyuan.aigc_extension.some_public_function(x)`
def some_public_function(x: int):
    """This is a public function that can be called from other extensions."""
    print(f"[hunyuan.aigc_extension] some_public_function was called with {x}")
    return x ** x


# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    """This extension manages AIGC 3D generation with image upload and deformable physics functionality."""
    
    def on_startup(self, ext_id):
        """This is called every time the extension is activated."""
        print("[hunyuan.aigc_extension] Extension startup")

        self._uploaded_images = []
        self._ext_id = ext_id
        self._selected_image_path = None
        self._generation_status = "Ready"
        self._generated_models = []
        self._pending_ui_updates = []
        self._auto_load_to_stage = True  # Auto-load generated models to stage
        self._keep_glb_files = False  # Keep GLB files after loading to stage
        self._apply_deformable_physics = False  # Apply deformable physics to loaded models
        self._apply_rigid_body_physics = False  # Apply rigid body physics
        
        # Deformable physics parameters
        self._simulation_resolution = 10
        self._youngs_modulus = 1e5  # Controls stiffness. Higher is stiffer.
        self._poissons_ratio = 0.45  # Controls volume preservation. 0.5 is incompressible.
        
        # Rigid body physics parameters
        self._collider_type = "convexHull"  # Default collider type
        self._mass_value = 10.0  # Default mass in kg
        self._use_density = False  # Use density instead of mass
        self._density_value = 1000.0  # Default density in kg/m³
        
        # Create directories
        self._images_dir = Path(__file__).parent / "uploaded_images"
        self._models_dir = Path(__file__).parent / "generated_models"
        self._images_dir.mkdir(exist_ok=True)
        self._models_dir.mkdir(exist_ok=True)
        
        # Hunyuan server configuration
        self._hunyuan_server_url = "http://localhost:8081"
        
        self._window = ui.Window(
            "AIGC Extension", width=450, height=800
        )
        
        with self._window.frame:
            # Add scrolling frame for all content
            main_scroll = ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
            )
            
            with main_scroll:
                with ui.VStack(spacing=10):
                    # Image upload section
                    ui.Label("Image Upload", height=30, style={"font_size": 18})
                    
                    with ui.HStack(height=30):
                        ui.Button("Select Image", clicked_fn=self._open_file_dialog, width=80)
                        ui.Button("Select Folder", clicked_fn=self._open_folder_dialog, width=80)
                        self._file_path_label = ui.Label("No file selected", word_wrap=True)
                    
                    # Image display area
                    ui.Label("Uploaded Images:", height=20)
                    
                    self._image_scroll = ui.ScrollingFrame(
                        height=200,
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
                    )
                    
                    with self._image_scroll:
                        self._image_stack = ui.VStack(spacing=5)
                    
                    # Clear all images button
                    ui.Button("Clear All Images", clicked_fn=self._clear_all_images, height=30)
                    
                    # Separator
                    ui.Separator(height=2)
                    
                    # 3D Generation section
                    ui.Label("3D Generation", height=30, style={"font_size": 18})
                    
                    # Selected image for generation
                    with ui.HStack(height=30):
                        ui.Label("Selected Image:", width=100)
                        self._selected_image_label = ui.Label("None selected", word_wrap=True)
                    
                    # Server configuration
                    with ui.HStack(height=30):
                        ui.Label("Server URL:", width=100)
                        self._server_url_field = ui.StringField()
                        self._server_url_field.model.set_value(self._hunyuan_server_url)
                    
                    # Auto-load option
                    with ui.HStack(height=25):
                        self._auto_load_checkbox = ui.CheckBox(width=20)
                        self._auto_load_checkbox.model.set_value(self._auto_load_to_stage)
                        self._auto_load_checkbox.model.add_value_changed_fn(self._on_auto_load_changed)
                        ui.Label("Auto-load to stage after generation")
                    
                    # Keep GLB files option
                    with ui.HStack(height=25):
                        self._keep_glb_checkbox = ui.CheckBox(width=20)
                        self._keep_glb_checkbox.model.set_value(self._keep_glb_files)
                        self._keep_glb_checkbox.model.add_value_changed_fn(self._on_keep_glb_changed)
                        ui.Label("Keep GLB files after loading (do not delete)")
                    
                    # Deformable physics option
                    if PHYSX_AVAILABLE:
                        with ui.HStack(height=25):
                            self._deformable_checkbox = ui.CheckBox(width=20)
                            self._deformable_checkbox.model.set_value(self._apply_deformable_physics)
                            self._deformable_checkbox.model.add_value_changed_fn(self._on_deformable_changed)
                            ui.Label("Apply deformable physics")
                        
                        # Physics parameters (only show when deformable is enabled)
                        self._physics_frame = ui.VStack(height=80)
                        with self._physics_frame:
                            ui.Label("Physics Parameters:", style={"font_size": 14})
                            with ui.VStack(spacing=5):
                                with ui.HStack(height=20):
                                    ui.Label("Resolution:", width=80)
                                    self._resolution_field = ui.IntField(width=60)
                                    self._resolution_field.model.set_value(self._simulation_resolution)
                                    self._resolution_field.model.add_value_changed_fn(self._on_resolution_changed)
                                
                                with ui.HStack(height=20):
                                    ui.Label("Stiffness:", width=80)
                                    self._stiffness_field = ui.FloatField(width=60)
                                    self._stiffness_field.model.set_value(self._youngs_modulus)
                                    self._stiffness_field.model.add_value_changed_fn(self._on_stiffness_changed)
                        
                        # Initially hide physics parameters
                        self._physics_frame.visible = self._apply_deformable_physics
                    else:
                        ui.Label("PhysX deformable not available", style={"color": 0xFFAA6600})
                    
                    # Separator
                    ui.Separator(height=2)
                    
                    # Rigid Body Physics option
                    ui.Label("Rigid Body Physics", height=25, style={"font_size": 16})
                    
                    with ui.HStack(height=25):
                        self._rigid_body_checkbox = ui.CheckBox(width=20)
                        self._rigid_body_checkbox.model.set_value(self._apply_rigid_body_physics)
                        self._rigid_body_checkbox.model.add_value_changed_fn(self._on_rigid_body_changed)
                        ui.Label("Apply rigid body collider")
                    
                    # Rigid body parameters
                    self._rigid_body_frame = ui.VStack(height=140)
                    with self._rigid_body_frame:
                        ui.Label("Collider & Mass Parameters:", style={"font_size": 14})
                        
                        with ui.VStack(spacing=5):
                            # Collider type selection
                            with ui.HStack(height=25):
                                ui.Label("Collider Type:", width=100)
                                self._collider_combo = ui.ComboBox(0, "Convex Hull", "Bounding Sphere", "Bounding Cube", 
                                                                   "Convex Decomposition", "Mesh Simplification")
                                self._collider_combo.model.add_item_changed_fn(self._on_collider_type_changed)
                            
                            # Mass type selection
                            with ui.HStack(height=25):
                                self._use_density_checkbox = ui.CheckBox(width=20)
                                self._use_density_checkbox.model.set_value(self._use_density)
                                self._use_density_checkbox.model.add_value_changed_fn(self._on_use_density_changed)
                                ui.Label("Use Density instead of Mass")
                            
                            # Mass value
                            with ui.HStack(height=25):
                                ui.Label("Mass (kg):", width=100)
                                self._mass_field = ui.FloatField(width=80)
                                self._mass_field.model.set_value(self._mass_value)
                                self._mass_field.model.add_value_changed_fn(self._on_mass_changed)
                                self._mass_field.enabled = not self._use_density
                            
                            # Density value
                            with ui.HStack(height=25):
                                ui.Label("Density (kg/m³):", width=100)
                                self._density_field = ui.FloatField(width=80)
                                self._density_field.model.set_value(self._density_value)
                                self._density_field.model.add_value_changed_fn(self._on_density_changed)
                                self._density_field.enabled = self._use_density
                    
                    # Initially hide rigid body parameters
                    self._rigid_body_frame.visible = self._apply_rigid_body_physics
                    
                    # Generation controls
                    with ui.HStack(height=40, spacing=10):
                        self._generate_btn = ui.Button("Generate 3D Model", clicked_fn=self._generate_3d_model, width=150)
                        self._test_server_btn = ui.Button("Test Server", clicked_fn=self._test_server_connection, width=100)
                    
                    # Status display
                    with ui.HStack(height=30):
                        ui.Label("Status:", width=60)
                        self._status_label = ui.Label("Ready", style={"color": 0xFF00AA00})
                    
                    # Generated models area
                    ui.Label("Generated Models:", height=20)
                    
                    self._models_scroll = ui.ScrollingFrame(
                        height=150,
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
                    )
                    
                    with self._models_scroll:
                        self._models_stack = ui.VStack(spacing=5)
                    
                    # Load external GLB and Clear all models buttons
                    with ui.HStack(height=30, spacing=10):
                        ui.Button("Load External GLB", clicked_fn=self._load_external_glb, height=30)
                        ui.Button("Clear All Models", clicked_fn=self._clear_all_models, height=30)
                    
                    # Separator
                    ui.Separator(height=2)
                    
                    # Scale Tool section
                    ui.Label("Scale Tool (Selected Prim)", height=30, style={"font_size": 18})
                    
                    ui.Label("Select a prim in viewport, enter actual height, then click Scale", word_wrap=True)
                    
                    # Selected prim display
                    with ui.HStack(height=25):
                        ui.Label("Selected:", width=80)
                        self._scale_selected_prim_label = ui.Label("None", word_wrap=True, style={"color": 0xFF888888})
                    
                    # Actual height input
                    with ui.HStack(height=30):
                        ui.Label("Actual Height:", width=100)
                        self._actual_height_field = ui.FloatField(width=100)
                        self._actual_height_field.model.set_value(1.0)
                        ui.Label("meters")
                    
                    # Height axis selection
                    with ui.HStack(height=30):
                        ui.Label("Height Axis:", width=100)
                        self._height_axis_combo = ui.ComboBox(0, "Y", "X", "Z", width=80)
                    
                    # Scale controls
                    with ui.HStack(height=35, spacing=10):
                        ui.Button("Get Selected Prim", clicked_fn=self._update_selected_prim_info, width=130)
                        ui.Button("Scale to Height", clicked_fn=self._scale_selected_prim, width=130)
                    
                    # Scale result display
                    with ui.HStack(height=25):
                        ui.Label("Scale Result:", width=100)
                        self._scale_result_label = ui.Label("Ready", style={"color": 0xFF00AA00})
        
        # Set up a timer to process pending UI updates from background threads
        self._setup_ui_update_timer()

    def _on_auto_load_changed(self, model):
        """Callback when auto-load checkbox state changes"""
        self._auto_load_to_stage = model.get_value_as_bool()
        print(f"Auto-load to stage: {'enabled' if self._auto_load_to_stage else 'disabled'}")
    
    def _on_keep_glb_changed(self, model):
        """Callback when keep GLB files checkbox state changes"""
        self._keep_glb_files = model.get_value_as_bool()
        print(f"Keep GLB files: {'enabled' if self._keep_glb_files else 'disabled'}")
    
    def _on_deformable_changed(self, model):
        """Callback when deformable physics checkbox state changes"""
        self._apply_deformable_physics = model.get_value_as_bool()
        print(f"Deformable physics: {'enabled' if self._apply_deformable_physics else 'disabled'}")
        
        # Show/hide physics parameters frame
        if hasattr(self, '_physics_frame'):
            self._physics_frame.visible = self._apply_deformable_physics
    
    def _on_resolution_changed(self, model):
        """Callback when simulation resolution changes"""
        self._simulation_resolution = model.get_value_as_int()
        print(f"Simulation resolution: {self._simulation_resolution}")
    
    def _on_stiffness_changed(self, model):
        """Callback when stiffness (Young's modulus) changes"""
        self._youngs_modulus = model.get_value_as_float()
        print(f"Young's modulus (stiffness): {self._youngs_modulus}")
    
    def _on_rigid_body_changed(self, model):
        """Callback when rigid body physics checkbox changes"""
        self._apply_rigid_body_physics = model.get_value_as_bool()
        print(f"Rigid body physics: {'enabled' if self._apply_rigid_body_physics else 'disabled'}")
        if hasattr(self, '_rigid_body_frame'):
            self._rigid_body_frame.visible = self._apply_rigid_body_physics
    
    def _on_collider_type_changed(self, model, item):
        """Callback when collider type changes"""
        collider_types = ["convexHull", "boundingSphere", "boundingCube", 
                         "convexDecomposition", "meshSimplification"]
        index = model.get_item_value_model().get_value_as_int()
        self._collider_type = collider_types[index]
        print(f"Collider type: {self._collider_type}")
    
    def _on_use_density_changed(self, model):
        """Callback when use density checkbox changes"""
        self._use_density = model.get_value_as_bool()
        self._mass_field.enabled = not self._use_density
        self._density_field.enabled = self._use_density
        print(f"Use density: {self._use_density}")
    
    def _on_mass_changed(self, model):
        """Callback when mass value changes"""
        self._mass_value = model.get_value_as_float()
        print(f"Mass: {self._mass_value} kg")
    
    def _on_density_changed(self, model):
        """Callback when density value changes"""
        self._density_value = model.get_value_as_float()
        print(f"Density: {self._density_value} kg/m³")

    def _setup_ui_update_timer(self):
        """Set up a timer to process UI updates from background threads"""
        import omni.kit.app
        
        # Create a timer that runs every 100ms to check for pending UI updates
        self._ui_update_timer = None
        
        def process_ui_updates():
            try:
                if self._pending_ui_updates:
                    # Process all pending updates
                    updates_to_process = self._pending_ui_updates[:]
                    self._pending_ui_updates.clear()
                    
                    for update in updates_to_process:
                        if update['type'] == 'add_model':
                            self._add_model_to_ui(update['model_path'], update['filename'])
                        elif update['type'] == 'auto_load_model':
                            self._auto_load_model_to_stage(update['model_path'], update['filename'])
            except Exception as e:
                print(f"Error processing UI updates: {e}")
        
        # Set up the timer using Omniverse's app framework
        app = omni.kit.app.get_app()
        self._ui_update_timer = app.get_update_event_stream().create_subscription_to_pop(
            lambda dt: process_ui_updates(), name="ui_update_timer"
        )

    def _open_file_dialog(self):
        """Open file selection dialog"""
        # Create a simple input window for file path
        input_window = ui.Window("Enter Image Path", width=400, height=150)
        
        with input_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Please enter the full path to the image file:")
                path_field = ui.StringField()
                path_field.model.set_value("/home/vcao/")  # Default path
                
                with ui.HStack():
                    def on_confirm():
                        path = path_field.model.get_value_as_string()
                        if path and os.path.exists(path) and self._is_valid_image(path):
                            self._upload_image(path)
                            input_window.visible = False
                        else:
                            self._file_path_label.text = "Error: File does not exist or is not a valid image file"
                    
                    def on_cancel():
                        input_window.visible = False
                        self._file_path_label.text = "Selection cancelled"
                    
                    ui.Button("Confirm", clicked_fn=on_confirm)
                    ui.Button("Cancel", clicked_fn=on_cancel)

    def _open_folder_dialog(self):
        """Open folder selection dialog"""
        # Create a simple input window for folder path
        input_window = ui.Window("Select Image Folder", width=500, height=180)
        
        with input_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Enter the full path to the folder containing images:")
                path_field = ui.StringField()
                path_field.model.set_value("/home/vcao/Pictures/")  # Default path
                
                ui.Label("Supported formats: PNG, JPG, JPEG")
                
                with ui.HStack():
                    def on_confirm():
                        folder_path = path_field.model.get_value_as_string()
                        if folder_path and os.path.exists(folder_path) and os.path.isdir(folder_path):
                            self._upload_images_from_folder(folder_path)
                            input_window.visible = False
                        else:
                            self._file_path_label.text = "Error: Folder does not exist"
                    
                    def on_cancel():
                        input_window.visible = False
                        self._file_path_label.text = "Folder selection cancelled"
                    
                    ui.Button("Load All Images", clicked_fn=on_confirm)
                    ui.Button("Cancel", clicked_fn=on_cancel)

    def _upload_images_from_folder(self, folder_path):
        """Upload all valid images from a folder"""
        try:
            # Get all files in the folder
            all_files = os.listdir(folder_path)
            
            # Filter for valid image files
            valid_extensions = ['.png', '.jpg', '.jpeg']
            image_files = []
            
            for file in all_files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in valid_extensions:
                    full_path = os.path.join(folder_path, file)
                    if os.path.isfile(full_path):
                        image_files.append(full_path)
            
            if not image_files:
                self._file_path_label.text = "No valid images found in folder"
                print(f"No valid images found in: {folder_path}")
                return
            
            print(f"Found {len(image_files)} image(s) in folder: {folder_path}")
            
            # Upload each image
            successful_uploads = 0
            failed_uploads = 0
            
            for image_path in image_files:
                try:
                    self._upload_single_image(image_path)
                    successful_uploads += 1
                    print(f"Uploaded: {os.path.basename(image_path)}")
                except Exception as e:
                    failed_uploads += 1
                    print(f"Failed to upload {os.path.basename(image_path)}: {e}")
            
            # Update status
            if successful_uploads > 0:
                self._file_path_label.text = f"Uploaded {successful_uploads} images from folder"
                if failed_uploads > 0:
                    self._file_path_label.text += f" ({failed_uploads} failed)"
            else:
                self._file_path_label.text = "Failed to upload any images from folder"
            
            print(f"[hunyuan.aigc_extension] Folder upload complete: {successful_uploads} success, {failed_uploads} failed")
            
        except Exception as e:
            self._file_path_label.text = f"Error reading folder: {str(e)}"
            print(f"[hunyuan.aigc_extension] Error reading folder {folder_path}: {e}")
    
    def _is_valid_image(self, file_path):
        """Check if file is a valid image file"""
        if not os.path.exists(file_path):
            return False
        
        if not os.path.isfile(file_path):
            return False
        
        valid_extensions = ['.png', '.jpg', '.jpeg']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in valid_extensions:
            return False
        
        # Additional check: try to verify it's actually an image
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def _upload_single_image(self, source_path):
        """Upload a single image file (used by both single file and folder upload)"""
        # Generate unique filename
        filename = os.path.basename(source_path)
        name, ext = os.path.splitext(filename)
        counter = 1
        
        while True:
            if counter == 1:
                new_filename = filename
            else:
                new_filename = f"{name}_{counter}{ext}"
            
            dest_path = self._images_dir / new_filename
            if not dest_path.exists():
                break
            counter += 1
        
        # Copy file to extension directory
        shutil.copy2(source_path, dest_path)
        
        # Add to image list
        self._uploaded_images.append(str(dest_path))
        
        # Update UI
        self._add_image_to_ui(str(dest_path), new_filename)
        
        return dest_path

    def _upload_image(self, source_path):
        """Upload and process image file (wrapper for single file upload)"""
        try:
            dest_path = self._upload_single_image(source_path)
            filename = os.path.basename(dest_path)
            
            # Update UI
            self._file_path_label.text = f"Uploaded: {filename}"
            
            print(f"[hunyuan.aigc_extension] Image uploaded: {dest_path}")
            
        except Exception as e:
            print(f"[hunyuan.aigc_extension] Image upload failed: {e}")
            self._file_path_label.text = f"Upload failed: {str(e)}"
    
    def _add_image_to_ui(self, image_path, filename):
        """Add image thumbnail to UI"""
        with self._image_stack:
            with ui.HStack(height=80, spacing=10):
                # Image thumbnail
                try:
                    image_widget = ui.Image(
                        image_path,
                        width=70,
                        height=70,
                        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT
                    )
                except Exception as e:
                    print(f"Cannot display image {image_path}: {e}")
                    image_widget = ui.Rectangle(width=70, height=70)
                    with image_widget:
                        ui.Label("Image\nLoad\nFailed", alignment=ui.Alignment.CENTER)
                
                # Image information
                with ui.VStack():
                    ui.Label(f"Filename: {filename}", word_wrap=True)
                    ui.Label(f"Path: {image_path}", word_wrap=True, style={"color": 0xFF888888})
                
                # Action buttons
                with ui.VStack(width=80):
                    def select_for_generation():
                        self._selected_image_path = image_path
                        self._selected_image_label.text = filename
                        print(f"Selected image for 3D generation: {filename}")
                    
                    def remove_image():
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                            if image_path in self._uploaded_images:
                                self._uploaded_images.remove(image_path)
                            # Clear selection if this image was selected
                            if self._selected_image_path == image_path:
                                self._selected_image_path = None
                                self._selected_image_label.text = "None selected"
                            # Rebuild UI
                            self._refresh_image_list()
                        except Exception as e:
                            print(f"Failed to delete image: {e}")
                    
                    ui.Button("Select", clicked_fn=select_for_generation, height=25)
                    ui.Button("Delete", clicked_fn=remove_image, height=25)
    
    def _refresh_image_list(self):
        """Refresh image list display"""
        # Clear existing image display
        self._image_stack.clear()
        
        # Re-add all images
        for image_path in self._uploaded_images:
            if os.path.exists(image_path):
                filename = os.path.basename(image_path)
                self._add_image_to_ui(image_path, filename)
    
    def _clear_all_images(self):
        """Clear all uploaded images"""
        try:
            # Delete all image files
            for image_path in self._uploaded_images:
                if os.path.exists(image_path):
                    os.remove(image_path)
            
            # Clear list
            self._uploaded_images.clear()
            
            # Clear UI display
            self._image_stack.clear()
            
            # Update status label
            self._file_path_label.text = "All images cleared"
            
            print("[hunyuan.aigc_extension] All images cleared")
            
        except Exception as e:
            print(f"[hunyuan.aigc_extension] Failed to clear images: {e}")
            self._file_path_label.text = f"Clear failed: {str(e)}"
    
    def _image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        try:
            with Image.open(image_path).convert("RGBA") as image:
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                return img_base64
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None
    
    def _test_server_connection(self):
        """Test connection to Hunyuan server"""
        try:
            server_url = self._server_url_field.model.get_value_as_string()
            self._update_status("Testing server connection...", "orange")
            
            # Test health endpoint
            response = requests.get(f"{server_url}/health", timeout=10)
            if response.status_code == 200:
                self._update_status("Server connection successful", "green")
                print(f"Server health check passed: {response.json()}")
            else:
                self._update_status(f"Server error: {response.status_code}", "red")
        except requests.exceptions.ConnectionError:
            self._update_status("Cannot connect to server", "red")
        except requests.exceptions.Timeout:
            self._update_status("Server connection timeout", "red")
        except Exception as e:
            self._update_status(f"Connection test failed: {str(e)}", "red")
    
    def _update_status(self, message, color="green"):
        """Update status label with message and color"""
        color_map = {
            "green": 0xFF00AA00,
            "red": 0xFFAA0000,
            "orange": 0xFFFF6600,
            "blue": 0xFF0066AA
        }
        self._status_label.text = message
        self._status_label.style = {"color": color_map.get(color, 0xFF00AA00)}
        print(f"[Status] {message}")
    
    def _generate_3d_model(self):
        """Generate 3D model from selected image"""
        if not self._selected_image_path:
            self._update_status("No image selected for generation", "red")
            return
        
        if not os.path.exists(self._selected_image_path):
            self._update_status("Selected image file not found", "red")
            return
        
        # Disable generate button during processing
        self._generate_btn.enabled = False
        
        # Start generation in a separate thread to avoid blocking UI
        thread = threading.Thread(target=self._generate_3d_model_async)
        thread.daemon = True
        thread.start()
    
    def _generate_3d_model_async(self):
        """Async method to generate 3D model"""
        try:
            server_url = self._server_url_field.model.get_value_as_string()
            
            self._update_status("Converting image to base64...", "blue")
            
            # Convert image to base64
            image_base64 = self._image_to_base64(self._selected_image_path)
            if not image_base64:
                self._update_status("Failed to process image", "red")
                self._generate_btn.enabled = True
                return
            
            # Prepare request data
            request_data = {
                "image": image_base64,
                "type": "glb"
            }
            
            self._update_status("Sending generation request...", "blue")
            
            # Send request to Hunyuan server
            response = requests.post(
                f"{server_url}/send", 
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                self._update_status(f"Server error: {response.status_code}", "red")
                self._generate_btn.enabled = True
                return
            
            result = response.json()
            uid = result.get("uid")
            
            if not uid:
                self._update_status("No task ID received", "red")
                self._generate_btn.enabled = True
                return
            
            self._update_status(f"Generation started (ID: {uid[:8]}...)", "blue")
            print(f"Task ID: {uid}")
            
            # Poll for completion
            self._poll_generation_status(server_url, uid)
            
        except requests.exceptions.ConnectionError:
            self._update_status("Cannot connect to server", "red")
            self._generate_btn.enabled = True
        except requests.exceptions.Timeout:
            self._update_status("Request timeout", "red")
            self._generate_btn.enabled = True
        except Exception as e:
            self._update_status(f"Generation failed: {str(e)}", "red")
            self._generate_btn.enabled = True
    
    def _poll_generation_status(self, server_url, uid):
        """Poll generation status until completion"""
        try:
            max_attempts = 120  # 10 minutes at 5-second intervals
            attempt = 0
            
            while attempt < max_attempts:
                # Check status
                status_response = requests.get(f"{server_url}/status/{uid}", timeout=10)
                
                if status_response.status_code != 200:
                    self._update_status("Failed to check status", "red")
                    break
                
                status_data = status_response.json()
                current_status = status_data.get('status', 'unknown')
                
                if current_status == 'completed':
                    self._update_status("Generation completed!", "green")
                    
                    # Save the generated model
                    model_base64 = status_data.get('model_base64')
                    if model_base64:
                        self._save_generated_model(uid, model_base64)
                    else:
                        self._update_status("No model data received", "red")
                    break
                    
                elif current_status == 'error':
                    error_msg = status_data.get('message', 'Unknown error')
                    self._update_status(f"Generation error: {error_msg}", "red")
                    break
                    
                elif current_status in ['processing', 'texturing']:
                    self._update_status(f"Status: {current_status}...", "blue")
                    
                elif current_status == 'pending':
                    self._update_status("Task queued, waiting...", "blue")
                
                attempt += 1
                time.sleep(5)  # Wait 5 seconds between checks
            
            if attempt >= max_attempts:
                self._update_status("Generation timeout", "red")
                
        except Exception as e:
            self._update_status(f"Status check failed: {str(e)}", "red")
        finally:
            self._generate_btn.enabled = True
    
    def _save_generated_model(self, uid, model_base64):
        """Save generated 3D model from base64 data"""
        try:
            # Generate filename
            timestamp = int(time.time())
            filename = f"model_{uid[:8]}_{timestamp}.glb"
            filepath = self._models_dir / filename
            
            # Decode and save
            model_data = base64.b64decode(model_base64)
            with open(filepath, 'wb') as f:
                f.write(model_data)
            
            # Add to generated models list
            self._generated_models.append(str(filepath))
            
            # Check if auto-load to stage is enabled
            if self._auto_load_to_stage:
                # Auto-load to stage and don't add to UI (since file will be deleted)
                self._pending_ui_updates.append({
                    'type': 'auto_load_model',
                    'model_path': str(filepath),
                    'filename': filename
                })
                self._update_status(f"Model saved and loading to stage: {filename}", "blue")
            else:
                # Just add to UI for manual loading later
                self._pending_ui_updates.append({
                    'type': 'add_model',
                    'model_path': str(filepath),
                    'filename': filename
                })
                self._update_status(f"Model saved: {filename}", "green")
            
            print(f"[hunyuan.aigc_extension] Model saved: {filepath}")
            
        except Exception as e:
            self._update_status(f"Failed to save model: {str(e)}", "red")
            print(f"[hunyuan.aigc_extension] Error saving model: {e}")

    def _auto_load_model_to_stage(self, model_path, filename):
        """Automatically load generated model to stage"""
        try:
            # Load the model to stage (this will also delete the file)
            self._load_glb_to_stage(model_path, filename)
            print(f"[hunyuan.aigc_extension] Auto-loaded model to stage: {filename}")
        except Exception as e:
            print(f"[hunyuan.aigc_extension] Failed to auto-load model to stage: {e}")
            # If auto-load fails, add to UI for manual loading
            self._add_model_to_ui(model_path, filename)

    def _add_model_to_ui(self, model_path, filename):
        """Add generated model to UI"""
        with self._models_stack:
            with ui.HStack(height=70, spacing=10):
                # Model icon (placeholder)
                model_icon = ui.Rectangle(width=40, height=40)
                with model_icon:
                    ui.Label("GLB", alignment=ui.Alignment.CENTER, style={"font_size": 12})
                
                # Model information
                with ui.VStack():
                    ui.Label(f"Model: {filename}", word_wrap=True)
                    ui.Label(f"Path: {model_path}", word_wrap=True, style={"color": 0xFF888888})
                
                # Action buttons
                with ui.VStack(width=100):
                    def load_to_stage():
                        try:
                            self._load_glb_to_stage(model_path, filename)
                        except Exception as e:
                            print(f"Failed to load to stage: {e}")
                            self._update_status(f"Failed to load to stage: {str(e)}", "red")
                    
                    def open_model_folder():
                        try:
                            # Open the folder containing the model
                            folder_path = os.path.dirname(model_path)
                            os.system(f"xdg-open '{folder_path}'")  # Linux
                        except Exception as e:
                            print(f"Failed to open folder: {e}")
                    
                    def delete_model():
                        try:
                            if os.path.exists(model_path):
                                os.remove(model_path)
                            if model_path in self._generated_models:
                                self._generated_models.remove(model_path)
                            # Rebuild UI
                            self._refresh_models_list()
                        except Exception as e:
                            print(f"Failed to delete model: {e}")
                    
                    ui.Button("Load to Stage", clicked_fn=load_to_stage, height=20)
                    with ui.HStack():
                        ui.Button("Open", clicked_fn=open_model_folder, height=20, width=45)
                        ui.Button("Delete", clicked_fn=delete_model, height=20, width=45)
    
    def _refresh_models_list(self):
        """Refresh generated models list display"""
        # Clear existing display
        self._models_stack.clear()
        
        # Re-add all models
        for model_path in self._generated_models:
            if os.path.exists(model_path):
                filename = os.path.basename(model_path)
                self._add_model_to_ui(model_path, filename)
    
    def _clear_all_models(self):
        """Clear all generated models"""
        try:
            # Delete all model files
            for model_path in self._generated_models:
                if os.path.exists(model_path):
                    os.remove(model_path)
            
            # Clear list
            self._generated_models.clear()
            
            # Clear UI display
            self._models_stack.clear()
            
            # Update status
            self._update_status("All models cleared", "green")
            
            print("[hunyuan.aigc_extension] All models cleared")
            
        except Exception as e:
            print(f"[hunyuan.aigc_extension] Failed to clear models: {e}")
            self._update_status(f"Clear failed: {str(e)}", "red")
    
    def _load_glb_to_stage(self, model_path, filename):
        """Load GLB model to Omniverse stage and delete the original file"""
        try:
            self._update_status(f"Loading {filename} to stage...", "blue")
            
            # Get current USD Stage
            stage = omni.usd.get_context().get_stage()
            if not stage:
                self._update_status("No USD stage available", "red")
                carb.log_error("Cannot get the current USD Stage. Please ensure you have opened a scene.")
                return
            
            carb.log_info(f"Successfully got the Stage: {stage}")
            
            # Create a unique prim path based on filename
            base_name = os.path.splitext(filename)[0]
            prim_path = f"/World/{base_name}"
            
            # Ensure unique prim path
            counter = 1
            original_prim_path = prim_path
            while stage.GetPrimAtPath(prim_path).IsValid():
                prim_path = f"{original_prim_path}_{counter}"
                counter += 1
            
            # Create Xform Prim
            prim = stage.DefinePrim(prim_path, "Xform")
            if not prim:
                self._update_status("Failed to create prim", "red")
                carb.log_error(f"Cannot create Prim at {prim_path}")
                return
            
            carb.log_info(f"Successfully created Prim: {prim.GetPath()}")
            
            # Add reference to GLB file
            prim.GetReferences().AddReference(model_path)
            carb.log_info(f"Successfully added reference to: {model_path}")
            
            # Set position at origin
            xformable = UsdGeom.Xformable(prim)
            
            # Check for existing transform operations and reuse them if they exist
            translate_op = None
            scale_op = None
            orient_op = None
            
            # Get existing operations
            existing_ops = xformable.GetOrderedXformOps()
            for op in existing_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op
            
            # Add translate operation if it doesn't exist
            if translate_op is None:
                translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(0, 0, 0))
            
            # Add scale operation if it doesn't exist
            if scale_op is None:
                scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
            scale_op.Set(Gf.Vec3f(1, 1, 1))
            
            # Add orientation operation if it doesn't exist
            if orient_op is None:
                orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
            orient_op.Set(Gf.Quatf(1, 0, 0, 0))  # W, X, Y, Z - identity quaternion
            
            carb.log_info(f"Successfully set the position of Prim '{prim_path}' to (0, 0, 0)")
            
            # Apply deformable physics if enabled
            if self._apply_deformable_physics:
                self._update_status(f"Applying deformable physics to {filename}...", "blue")
                physics_success = self._apply_deformable_physics_to_prim(prim_path, os.path.splitext(filename)[0])
                if physics_success:
                    self._update_status(f"Deformable physics applied to {filename}", "green")
                else:
                    self._update_status(f"Deformable physics failed for {filename}", "orange")
            
            # Apply rigid body collider if enabled
            if self._apply_rigid_body_physics:
                self._update_status(f"Applying rigid body collider to {filename}...", "blue")
                collider_success = self._apply_rigid_body_collider(prim_path, os.path.splitext(filename)[0])
                if collider_success:
                    self._update_status(f"Rigid body collider applied to {filename}", "green")
                else:
                    self._update_status(f"Rigid body collider failed for {filename}", "orange")
            
            # Delete the original GLB file after successful loading (unless user wants to keep it)
            if not self._keep_glb_files:
                try:
                    os.remove(model_path)
                    carb.log_info(f"Deleted original GLB file: {model_path}")
                    
                    # Remove from generated models list
                    if model_path in self._generated_models:
                        self._generated_models.remove(model_path)
                    
                    # Refresh the models UI
                    self._refresh_models_list()
                    
                    self._update_status(f"Loaded {filename} to stage and cleaned up file", "green")
                    
                except Exception as delete_error:
                    carb.log_warn(f"Failed to delete original GLB file: {delete_error}")
                    self._update_status(f"Loaded {filename} to stage (file cleanup failed)", "orange")
            else:
                # Keep the file
                carb.log_info(f"Keeping GLB file: {model_path}")
                self._update_status(f"Loaded {filename} to stage (file kept)", "green")
            
            carb.log_info("GLB loading to stage completed successfully")
            
        except Exception as e:
            self._update_status(f"Failed to load to stage: {str(e)}", "red")
            carb.log_error(f"Error loading GLB to stage: {e}")
            print(f"[hunyuan.aigc_extension] Error loading GLB to stage: {e}")
    
    def _load_external_glb(self):
        """Load external GLB file to stage"""
        # Create a simple input window for GLB path
        input_window = ui.Window("Load External GLB", width=500, height=200)
        
        with input_window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Enter the full path to the GLB file:")
                path_field = ui.StringField()
                path_field.model.set_value("/home/vcao/")  # Default path
                
                ui.Label("Enter the name for the object in the stage:")
                name_field = ui.StringField()
                name_field.model.set_value("external_model")  # Default name
                
                with ui.HStack():
                    def on_confirm():
                        glb_path = path_field.model.get_value_as_string()
                        model_name = name_field.model.get_value_as_string()
                        
                        if glb_path and os.path.exists(glb_path) and glb_path.lower().endswith('.glb'):
                            try:
                                self._load_glb_to_stage_direct(glb_path, model_name)
                                input_window.visible = False
                            except Exception as e:
                                self._update_status(f"Failed to load GLB: {str(e)}", "red")
                        else:
                            self._update_status("Invalid GLB file path", "red")
                    
                    def on_cancel():
                        input_window.visible = False
                    
                    ui.Button("Load to Stage", clicked_fn=on_confirm)
                    ui.Button("Cancel", clicked_fn=on_cancel)
    
    def _load_glb_to_stage_direct(self, glb_path, model_name):
        """Load GLB file directly to stage without deleting the original"""
        try:
            self._update_status(f"Loading {model_name} to stage...", "blue")
            
            # Get current USD Stage
            stage = omni.usd.get_context().get_stage()
            if not stage:
                self._update_status("No USD stage available", "red")
                carb.log_error("Cannot get the current USD Stage. Please ensure you have opened a scene.")
                return
            
            carb.log_info(f"Successfully got the Stage: {stage}")
            
            # Create a unique prim path based on model name
            prim_path = f"/World/{model_name}"
            
            # Ensure unique prim path
            counter = 1
            original_prim_path = prim_path
            while stage.GetPrimAtPath(prim_path).IsValid():
                prim_path = f"{original_prim_path}_{counter}"
                counter += 1
            
            # Create Xform Prim
            prim = stage.DefinePrim(prim_path, "Xform")
            if not prim:
                self._update_status("Failed to create prim", "red")
                carb.log_error(f"Cannot create Prim at {prim_path}")
                return
            
            carb.log_info(f"Successfully created Prim: {prim.GetPath()}")
            
            # Add reference to GLB file
            prim.GetReferences().AddReference(glb_path)
            carb.log_info(f"Successfully added reference to: {glb_path}")
            
            # Set position at origin
            xformable = UsdGeom.Xformable(prim)
            
            # Check for existing transform operations and reuse them if they exist
            translate_op = None
            scale_op = None
            orient_op = None
            
            # Get existing operations
            existing_ops = xformable.GetOrderedXformOps()
            for op in existing_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op
            
            # Add translate operation if it doesn't exist
            if translate_op is None:
                translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(0, 0, 0))
            
            # Add scale operation if it doesn't exist
            if scale_op is None:
                scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
            scale_op.Set(Gf.Vec3f(1, 1, 1))
            
            # Add orientation operation if it doesn't exist
            if orient_op is None:
                orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
            orient_op.Set(Gf.Quatf(1, 0, 0, 0))  # W, X, Y, Z - identity quaternion
            
            carb.log_info(f"Successfully set the position of Prim '{prim_path}' to (0, 0, 0)")
            
            # Apply deformable physics if enabled
            if self._apply_deformable_physics:
                self._update_status(f"Applying deformable physics to {model_name}...", "blue")
                physics_success = self._apply_deformable_physics_to_prim(prim_path, model_name)
                if physics_success:
                    self._update_status(f"Deformable physics applied to {model_name}", "green")
                else:
                    self._update_status(f"Deformable physics failed for {model_name}", "orange")
            
            # Apply rigid body collider if enabled
            if self._apply_rigid_body_physics:
                self._update_status(f"Applying rigid body collider to {model_name}...", "blue")
                collider_success = self._apply_rigid_body_collider(prim_path, model_name)
                if collider_success:
                    self._update_status(f"Rigid body collider applied to {model_name}", "green")
                else:
                    self._update_status(f"Rigid body collider failed for {model_name}", "orange")
            
            if not self._apply_deformable_physics and not self._apply_rigid_body_physics:
                self._update_status(f"Successfully loaded {model_name} to stage", "green")
            
            carb.log_info("External GLB loading to stage completed successfully")
            
        except Exception as e:
            self._update_status(f"Failed to load external GLB: {str(e)}", "red")
            carb.log_error(f"Error loading external GLB to stage: {e}")
            print(f"[hunyuan.aigc_extension] Error loading external GLB to stage: {e}")
    
    def _find_mesh_prims(self, root_prim):
        """Recursively find all mesh prims under a root prim"""
        mesh_prims = []
        
        def traverse_prim(prim):
            # Check if this prim is a mesh
            if prim.GetTypeName() == 'Mesh':
                mesh_prims.append(prim)
            
            # Recursively check children
            for child in prim.GetChildren():
                traverse_prim(child)
        
        traverse_prim(root_prim)
        return mesh_prims
    
    def _apply_rigid_body_collider(self, prim_path_str, model_name):
        """Apply rigid body collider and mass to the root prim and its mesh children"""
        try:
            stage = omni.usd.get_context().get_stage()
            if not stage:
                carb.log_error("No USD stage available")
                return False
            
            root_prim = stage.GetPrimAtPath(prim_path_str)
            if not root_prim.IsValid():
                carb.log_error(f"Prim not found: {prim_path_str}")
                return False
            
            carb.log_info(f"Applying rigid body physics to: {prim_path_str}")
            
            # 参考代码: 在根Xform上设置transform操作和RigidBodyAPI
            rigidBodyXform = UsdGeom.Xform(root_prim)
            if not rigidBodyXform:
                carb.log_error(f"Cannot create Xform for: {prim_path_str}")
                return False
            
            # 添加或获取transform操作
            translate_op = None
            orient_op = None
            existing_ops = rigidBodyXform.GetOrderedXformOps()
            for op in existing_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op
            
            # 如果不存在则添加translate操作
            if translate_op is None:
                translate_op = rigidBodyXform.AddTranslateOp()
                translate_op.Set(Gf.Vec3f(0, 0, 0))
            
            # 如果不存在则添加orient操作
            if orient_op is None:
                orient_op = rigidBodyXform.AddOrientOp()
                orient_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # 单位四元数 (w, x, y, z)
            
            carb.log_info(f"Transform operations set for: {prim_path_str}")
            
            # 应用RigidBodyAPI到根prim
            rigidBodyPrim = rigidBodyXform.GetPrim()
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(rigidBodyPrim)
            carb.log_info(f"Applied RigidBodyAPI to: {prim_path_str}")
            
            # 应用MassAPI并设置mass或density
            massAPI = UsdPhysics.MassAPI.Apply(rigidBodyPrim)
            
            if self._use_density:
                # 使用CreateDensityAttr设置密度（参考代码方式）
                massAPI.CreateDensityAttr(self._density_value)
                carb.log_info(f"Set density: {self._density_value} kg/m³")
            else:
                # 使用CreateMassAttr设置质量
                massAPI.CreateMassAttr(self._mass_value)
                carb.log_info(f"Set mass: {self._mass_value} kg")
            
            # 在子mesh上应用collider
            mesh_prims = self._find_mesh_prims(root_prim)
            if not mesh_prims:
                carb.log_warn(f"No mesh prims found under {prim_path_str}")
                # 即使没有mesh，如果RigidBodyAPI已应用，仍然返回True
                return True
            
            carb.log_info(f"Found {len(mesh_prims)} mesh prim(s) for collider")
            
            success_count = 0
            for mesh_prim in mesh_prims:
                try:
                    mesh_path = mesh_prim.GetPath()
                    carb.log_info(f"Applying collider to mesh: {mesh_path}")
                    
                    # Apply CollisionAPI
                    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
                    
                    # Apply MeshCollisionAPI with approximation
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
                    
                    # Set collider type
                    collider_token = getattr(UsdPhysics.Tokens, self._collider_type)
                    mesh_collision_api.GetApproximationAttr().Set(collider_token)
                    
                    carb.log_info(f"Applied {self._collider_type} collider to: {mesh_path}")
                    
                    success_count += 1
                    
                except Exception as e:
                    carb.log_error(f"Failed to apply collider to {mesh_path}: {e}")
                    continue
            
            if success_count > 0:
                carb.log_info(f"Successfully applied collider to {success_count}/{len(mesh_prims)} meshes")
            
            carb.log_info(f"Rigid body physics setup complete for: {prim_path_str}")
            return True
            
        except Exception as e:
            carb.log_error(f"Error applying rigid body collider: {e}")
            print(f"[hunyuan.aigc_extension] Error applying rigid body collider: {e}")
            return False

    def _apply_deformable_physics_to_prim(self, prim_path_str, model_name):
        """Apply deformable physics to mesh prims under the given path"""
        if not PHYSX_AVAILABLE:
            carb.log_warn("PhysX deformable utilities not available")
            return False
        
        try:
            stage = omni.usd.get_context().get_stage()
            if not stage:
                carb.log_error("No USD stage available for physics application")
                return False
            
            root_prim_path = Sdf.Path(prim_path_str)
            root_prim = stage.GetPrimAtPath(root_prim_path)
            
            if not root_prim.IsValid():
                carb.log_error(f"Root prim not found at path: {prim_path_str}")
                return False
            
            carb.log_info(f"Searching for mesh prims under: {prim_path_str}")
            
            # Find all mesh prims under the root prim
            mesh_prims = self._find_mesh_prims(root_prim)
            
            if not mesh_prims:
                carb.log_warn(f"No mesh prims found under {prim_path_str}")
                return False
            
            carb.log_info(f"Found {len(mesh_prims)} mesh prim(s) to apply physics to:")
            for mesh_prim in mesh_prims:
                carb.log_info(f"  - {mesh_prim.GetPath()}")
            
            # Apply deformable physics to each mesh prim
            success_count = 0
            for i, mesh_prim in enumerate(mesh_prims):
                try:
                    mesh_path = mesh_prim.GetPath()
                    carb.log_info(f"Applying deformable physics to mesh: {mesh_path}")
                    
                    # Apply deformable body properties to the mesh
                    deformable_body = deformableUtils.add_physx_deformable_body(
                        stage,
                        mesh_path,
                        collision_simplification=True,
                        simulation_hexahedral_resolution=self._simulation_resolution,
                        self_collision=False,
                    )
                    
                    carb.log_info(f"Deformable body applied successfully to: {mesh_path}")
                    
                    # Create deformable physics material for this mesh
                    carb.log_info(f"Creating deformable material for mesh {i+1}...")
                    
                    # Create a unique path for the new material
                    material_path = omni.usd.get_stage_next_free_path(
                        stage, f"/World/PhysicsMaterial/DeformableMaterial_{model_name}_mesh_{i+1}", True
                    )
                    
                    # Create material with deformable properties
                    deformableUtils.add_deformable_body_material(
                        stage,
                        material_path,
                        youngs_modulus=self._youngs_modulus,
                        poissons_ratio=self._poissons_ratio,
                        damping_scale=0.0,
                        dynamic_friction=0.5,
                    )
                    carb.log_info(f"Deformable material created at: {material_path}")
                    
                    # Bind the material to the mesh prim
                    physicsUtils.add_physics_material_to_prim(stage, mesh_prim, material_path)
                    carb.log_info(f"Material bound to mesh: {mesh_path}")
                    
                    success_count += 1
                    
                except Exception as mesh_error:
                    carb.log_error(f"Failed to apply physics to mesh {mesh_path}: {mesh_error}")
                    continue
            
            if success_count > 0:
                carb.log_info(f"Successfully applied deformable physics to {success_count}/{len(mesh_prims)} meshes")
                return True
            else:
                carb.log_error("Failed to apply physics to any mesh prims")
                return False
            
        except Exception as e:
            carb.log_error(f"Error applying deformable physics: {e}")
            print(f"[hunyuan.aigc_extension] Error applying deformable physics: {e}")
            return False
    
    # ============================================================================
    # Scale Tool Functions
    # ============================================================================
    
    def _get_selected_prim_path(self):
        """
        Get the currently selected prim path from the stage
        
        Returns:
            String of selected prim path, or None if no selection
        """
        try:
            context = omni.usd.get_context()
            selection = context.get_selection()
            selected_paths = selection.get_selected_prim_paths()
            
            if not selected_paths:
                carb.log_warn("No prim selected!")
                return None
            
            # Return the first selected prim
            prim_path = selected_paths[0]
            carb.log_info(f"Selected prim: {prim_path}")
            return prim_path
        except Exception as e:
            carb.log_error(f"Error getting selected prim: {e}")
            return None
    
    def _update_selected_prim_info(self):
        """Update the UI with currently selected prim information"""
        prim_path = self._get_selected_prim_path()
        
        if not prim_path:
            self._scale_selected_prim_label.text = "None (Please select a prim)"
            self._scale_result_label.text = "No prim selected"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            return
        
        # Get stage
        stage = omni.usd.get_context().get_stage()
        if not stage:
            self._scale_result_label.text = "No stage available"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            return
        
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            self._scale_result_label.text = "Invalid prim"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            return
        
        # Get bounding box info
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.GetRange()
        
        if bbox_range.IsEmpty():
            self._scale_result_label.text = "Empty bounding box"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            return
        
        dimensions = bbox_range.GetMax() - bbox_range.GetMin()
        
        # Display prim info
        prim_name = prim_path.split('/')[-1]
        self._scale_selected_prim_label.text = f"{prim_name} ({prim_path})"
        
        # Display current dimensions
        height_axis_idx = self._height_axis_combo.model.get_item_value_model().as_int
        axis_names = ['Y', 'X', 'Z']
        height_axis = axis_names[height_axis_idx]
        
        if height_axis == 'Z':
            current_height = dimensions[2]
        elif height_axis == 'X':
            current_height = dimensions[0]
        else:  # Y
            current_height = dimensions[1]
        
        self._scale_result_label.text = f"Current size: W={dimensions[0]:.3f}, H({height_axis})={current_height:.3f}, D={dimensions[2]:.3f}m"
        self._scale_result_label.style = {"color": 0xFF00AA00}
        
        carb.log_info("=" * 70)
        carb.log_info(f"Selected Prim Information:")
        carb.log_info(f"  Path: {prim_path}")
        carb.log_info(f"  Dimensions (X, Y, Z): ({dimensions[0]:.6f}, {dimensions[1]:.6f}, {dimensions[2]:.6f}) meters")
        carb.log_info(f"  Current Height ({height_axis} axis): {current_height:.6f} meters")
        carb.log_info("=" * 70)
    
    def _scale_selected_prim(self):
        """Scale the currently selected prim to the specified actual height"""
        # Get selected prim
        prim_path = self._get_selected_prim_path()
        
        if not prim_path:
            self._scale_result_label.text = "Failed: No prim selected"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            carb.log_error("No prim selected. Please select a prim in the viewport first.")
            return
        
        # Get stage
        stage = omni.usd.get_context().get_stage()
        if not stage:
            self._scale_result_label.text = "Failed: No stage available"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            carb.log_error("No USD stage available")
            return
        
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            self._scale_result_label.text = "Failed: Invalid prim"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            carb.log_error(f"Invalid prim path: {prim_path}")
            return
        
        # Get actual height from UI
        actual_height = self._actual_height_field.model.get_value_as_float()
        
        if actual_height <= 0:
            self._scale_result_label.text = "Failed: Height must be > 0"
            self._scale_result_label.style = {"color": 0xFFFF4444}
            carb.log_error(f"Invalid height value: {actual_height}")
            return
        
        # Get height axis from combo box
        height_axis_idx = self._height_axis_combo.model.get_item_value_model().as_int
        axis_names = ['Y', 'X', 'Z']
        height_axis = axis_names[height_axis_idx]
        
        # Perform scaling
        result = self._scale_prim_to_height(prim_path, stage, actual_height, height_axis)
        
        if result and result['scaled']:
            scale_factor = result['scale_factor']
            self._scale_result_label.text = f"✓ Scaled {scale_factor:.4f}x to {actual_height:.3f}m"
            self._scale_result_label.style = {"color": 0xFF00AA00}
        elif result and not result['scaled']:
            self._scale_result_label.text = f"Height already correct ({actual_height:.3f}m)"
            self._scale_result_label.style = {"color": 0xFFAAAA00}
        else:
            self._scale_result_label.text = "Failed to scale prim"
            self._scale_result_label.style = {"color": 0xFFFF4444}
    
    def _scale_prim_to_height(self, prim_path, stage, actual_height, height_axis='Y', tolerance=0.001):
        """
        Scale a prim to match the specified actual height with uniform scaling
        
        Args:
            prim_path: USD prim path
            stage: USD stage
            actual_height: Target height in meters
            height_axis: Which axis represents height ('X', 'Y', or 'Z')
            tolerance: Skip scaling if within tolerance
            
        Returns:
            Dict with scaling information, or None if failed
        """
        try:
            prim = stage.GetPrimAtPath(prim_path)
            
            # Get current world space dimensions
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.GetRange()
            
            if bbox_range.IsEmpty():
                carb.log_error(f"Prim '{prim_path}' has empty bounding box")
                return None
            
            min_point = bbox_range.GetMin()
            max_point = bbox_range.GetMax()
            dimensions = max_point - min_point
            
            # Determine height based on specified axis
            height_axis = height_axis.upper()
            if height_axis == 'Z':
                height_idx = 2
                current_width = dimensions[0]
                current_height = dimensions[2]
                current_depth = dimensions[1]
                axis_labels = ('X (width)', 'Z (height)', 'Y (depth)')
            elif height_axis == 'X':
                height_idx = 0
                current_width = dimensions[1]
                current_height = dimensions[0]
                current_depth = dimensions[2]
                axis_labels = ('Y (width)', 'X (height)', 'Z (depth)')
            else:  # Default Y
                height_idx = 1
                current_width = dimensions[0]
                current_height = dimensions[1]
                current_depth = dimensions[2]
                axis_labels = ('X (width)', 'Y (height)', 'Z (depth)')
            
            # Get current local scale
            xformable = UsdGeom.Xformable(prim)
            current_scale = Gf.Vec3d(1.0, 1.0, 1.0)
            scale_op = None
            
            for xform_op in xformable.GetOrderedXformOps():
                if xform_op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    current_scale = xform_op.Get()
                    scale_op = xform_op
                    break
            
            # Calculate scale factor
            scale_factor = actual_height / current_height
            
            # Calculate new scale (uniform scaling)
            new_scale = Gf.Vec3d(
                current_scale[0] * scale_factor,
                current_scale[1] * scale_factor,
                current_scale[2] * scale_factor
            )
            
            # Calculate expected new dimensions
            new_width = current_width * scale_factor
            new_height = current_height * scale_factor
            new_depth = current_depth * scale_factor
            
            # Display detailed information
            carb.log_info("=" * 80)
            carb.log_info("SCALE TO ACTUAL HEIGHT")
            carb.log_info("=" * 80)
            carb.log_info(f"Selected Prim: {prim_path}")
            carb.log_info(f"Height Axis: {height_axis}")
            carb.log_info("")
            
            carb.log_info("CURRENT MODEL DIMENSIONS (World Space, meters):")
            carb.log_info(f"  {axis_labels[0]}: {current_width:.6f} m")
            carb.log_info(f"  {axis_labels[1]}: {current_height:.6f} m  ← Current Height")
            carb.log_info(f"  {axis_labels[2]}: {current_depth:.6f} m")
            carb.log_info(f"  Raw (X, Y, Z): ({dimensions[0]:.6f}, {dimensions[1]:.6f}, {dimensions[2]:.6f})")
            carb.log_info("")
            
            carb.log_info("CURRENT LOCAL SCALE:")
            carb.log_info(f"  Scale (X, Y, Z): ({current_scale[0]:.6f}, {current_scale[1]:.6f}, {current_scale[2]:.6f})")
            carb.log_info("")
            
            carb.log_info("TARGET HEIGHT:")
            carb.log_info(f"  Target Height ({height_axis}): {actual_height:.6f} m")
            carb.log_info(f"  Current Height ({height_axis}): {current_height:.6f} m")
            carb.log_info(f"  Difference: {abs(current_height - actual_height):.6f} m")
            carb.log_info("")
            
            carb.log_info("REQUIRED SCALE FACTOR:")
            carb.log_info(f"  Scale Factor: {scale_factor:.6f}x ({scale_factor * 100:.2f}%)")
            if scale_factor > 1.0:
                carb.log_info(f"  Action: Scale UP by {(scale_factor - 1.0) * 100:.2f}%")
            elif scale_factor < 1.0:
                carb.log_info(f"  Action: Scale DOWN by {(1.0 - scale_factor) * 100:.2f}%")
            carb.log_info("")
            
            # Check if height is already correct
            height_diff = abs(current_height - actual_height)
            if height_diff <= tolerance:
                carb.log_info("RESULT:")
                carb.log_info(f"  ✓ Height already correct! Current: {current_height:.6f}m, Target: {actual_height:.6f}m")
                carb.log_info(f"  Difference {height_diff:.6f}m <= Tolerance {tolerance:.6f}m")
                carb.log_info("  Skipping scaling.")
                carb.log_info("=" * 80)
                
                return {
                    'prim_path': prim_path,
                    'current_dimensions': {'width': current_width, 'height': current_height, 'depth': current_depth},
                    'current_scale': current_scale,
                    'target_height': actual_height,
                    'scale_factor': scale_factor,
                    'new_scale': new_scale,
                    'scaled': False,
                    'reason': 'Height already correct'
                }
            
            # Warn if scale factor is extreme
            if scale_factor > 10 or scale_factor < 0.1:
                carb.log_warn("⚠ WARNING: Extreme scale factor!")
                carb.log_warn(f"  Scale factor: {scale_factor:.6f}x")
                carb.log_warn("  This might indicate unit mismatch or incorrect values!")
            
            carb.log_info("UNIFORM SCALING (All axes scaled equally):")
            carb.log_info(f"  New Local Scale (X, Y, Z): ({new_scale[0]:.6f}, {new_scale[1]:.6f}, {new_scale[2]:.6f})")
            carb.log_info(f"  Expected New Dimensions:")
            carb.log_info(f"    {axis_labels[0]}: {new_width:.6f} m (change: {new_width - current_width:+.6f})")
            carb.log_info(f"    {axis_labels[1]}: {new_height:.6f} m (change: {new_height - current_height:+.6f})")
            carb.log_info(f"    {axis_labels[2]}: {new_depth:.6f} m (change: {new_depth - current_depth:+.6f})")
            carb.log_info("")
            
            # Apply scaling
            if scale_op is None:
                carb.log_info("Creating new scale operation...")
                scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
            else:
                carb.log_info("Updating existing scale operation...")
            
            scale_op.Set(new_scale)
            
            carb.log_info("RESULT:")
            carb.log_info(f"  ✓ Successfully scaled prim!")
            carb.log_info(f"  Target Height: {actual_height:.6f} m")
            carb.log_info(f"  Scale Factor: {scale_factor:.6f}x")
            carb.log_info(f"  New Scale: ({new_scale[0]:.6f}, {new_scale[1]:.6f}, {new_scale[2]:.6f})")
            carb.log_info("=" * 80)
            
            return {
                'prim_path': prim_path,
                'current_dimensions': {'width': current_width, 'height': current_height, 'depth': current_depth},
                'new_dimensions': {'width': new_width, 'height': new_height, 'depth': new_depth},
                'current_scale': current_scale,
                'new_scale': new_scale,
                'target_height': actual_height,
                'scale_factor': scale_factor,
                'height_axis': height_axis,
                'scaled': True,
                'reason': 'Success'
            }
            
        except Exception as e:
            carb.log_error(f"Failed to scale prim: {e}")
            return None

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        print("[hunyuan.aigc_extension] Extension shutdown")
        
        # Clean up the UI update timer
        if hasattr(self, '_ui_update_timer') and self._ui_update_timer:
            self._ui_update_timer.unsubscribe()
            self._ui_update_timer = None
        
        # Optional: clean up temporary image files on shutdown
        # self._clear_all_images() 