# AIGC Extension Overview

## Introduction

The AIGC Extension is a powerful tool for generating 3D models from 2D images using AI-powered generation services.

## Architecture

The extension consists of:
- **UI Layer**: Built with omni.ui for user interaction
- **API Layer**: Communicates with Hunyuan generation server
- **Physics Integration**: Supports both deformable and rigid body physics
- **Asset Management**: Handles image uploads and model storage

## Technical Details

### Image Processing
- Supports JPG, JPEG, PNG formats
- Automatic image compression and base64 encoding
- Organized storage in `uploaded_images/` directory

### 3D Generation
- Asynchronous generation using threading
- Progress tracking and status updates
- Automatic model saving to `generated_models/` directory

### Physics Features
- **Deformable Physics**: For soft body simulation
  - Customizable simulation resolution
  - Young's modulus and Poisson's ratio parameters
- **Rigid Body Physics**: For solid objects
  - Multiple collider types (convex hull, mesh, sphere, cube, etc.)
  - Mass and density configuration


