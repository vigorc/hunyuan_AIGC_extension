# AIGC Extension

## Overview

This extension provides AIGC 3D generation functionality with image upload, deformable physics support, and scale tools for NVIDIA Omniverse.

## Features

- **Image Upload**: Upload single images or entire folders
- **3D Generation**: Generate 3D models from images using Hunyuan API
- **Physics Support**: 
  - Deformable physics with customizable parameters
  - Rigid body physics with multiple collider types
- **Auto-loading**: Automatically load generated models to stage
- **Scale Tool**: Scale generated models to actual real-world dimensions

## Usage

### 3D Generation
1. Select an image using "Select Image" or "Select Folder"
2. Configure server URL (default: http://localhost:8081)
3. Click "Generate 3D Model" to start generation
4. Generated models will be saved and optionally loaded to the stage

### Scale Tool
1. Select a prim (model) in the viewport
2. Click "Get Selected Prim" to see current dimensions
3. Enter the actual height of the object in meters
4. Select the height axis (Y, X, or Z - default is Y)
5. Click "Scale to Height" to apply uniform scaling

The scale tool will:
- Display current model dimensions in world space
- Calculate required scale factor
- Show the scale factor (e.g., 1.5x means 150% of original size)
- Apply uniform scaling to match actual height
- Output detailed information to console

**Example**: If you generated a chair that shows 2.0m height but the actual chair is 0.95m:
- Current: 2.0m
- Target: 0.95m
- Scale Factor: 0.475x (47.5%, scaled DOWN by 52.5%)
- Result: Chair will be uniformly scaled to 0.95m height

## Requirements

- NVIDIA Omniverse
- Hunyuan API server running on configured URL (for 3D generation)


