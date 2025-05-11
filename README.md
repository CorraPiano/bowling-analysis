# Trajectory and Spin Detection in Bowling Throws

<div align="center">
   <img src="https://img.shields.io/badge/Language-Python-brightgreen" alt="Language" />
   <img src="https://img.shields.io/badge/Platform-VSCode-blue" alt="Platform" />
   <img src="https://img.shields.io/badge/Powered_by-OpenCV-red" alt="By" />
   <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" />
</div>

## Project Overview

This project focuses on detecting the **trajectory** and **spin** of a bowling ball using computer vision techniques. Leveraging Python, OpenCV, and 3D reconstruction, the system processes video recordings to extract detailed motion information for performance analysis or further research.

## Video results

Example outputs from the analysis pipeline:

<video controls src="data/recording_2/Final_2.mp4" title="Title"></video>
<video controls src="data/recording_3/Final_3.mp4" title="Title"></video>

## Techniques Used

- Object detection (ball tracking) using OpenCV
- Frame extraction and video preprocessing
- Lane and background segmentation
- 3D trajectory reconstruction
- Spin and rotation estimation
- Visualization and result compilation

## Repository structure

```bash
├── data/                      # Raw and processed data
│   ├── auxiliary_data/        # Shared resources across modules
│   ├── recording_1/           # Data for recording 1
│   ├── recording_2/           # Data for recording 2
│   │   ├── frames/                    
│   │   ├── Output_detected_test.mp4    
│   │   ├── Recording_2_normal_speed.mp4
│   │   ├── Recording_2_slow_motion.mp4
│   │   ├── Tracked_output.mp4
│   ├── recording_3/           # Data for recording 3
│   └── ...                    # Additional recordings

├── documents/                 # Notes and reference materials
│   ├── papers/                # Research papers
│   │   └── AReal-TimeBallDetectionApproachUsingCNN.pdf
│   ├── Links.txt             
│   ├── Notes.txt             
│   └── Repository_structure.txt 

├── notebook/                  # Jupyter notebooks
│   ├── ball_detection/        
│   ├── lane_detection/        
│   ├── reconstruction/        
│   ├── spin/                  
│   ├── trajectory/            
│   ├── various_tests/         
│   └── video_creation/        

├── src/                       # Source code
│   ├── main.py                # Project entry point
│   ├── ball_detection/        # Ball detection logic
│   │   └── Detection.py       
│   ├── lane_detection/        
│   ├── reconstruction/        
│   ├── spin/                  
│   ├── trajectory/            
│   └── utility/               # Utility functions
```

## Hot to run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bowling-trajectory-spin.git
   cd bowling-trajectory-spin

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
    ```
3. Navigate to the `src` directory:
   ```bash
   cd src
   ```
4. Run the main script:
   ```bash
   python main.py
   ```
5. Follow the prompts to select the recording and adjust parameters as needed.
6. Use Jupyter notebooks in the `notebook/` directory for analysis and visualization.

## Contacts

For questions or collaboration inquiries, feel free to reach out:

- Davide Corradina – Mail: davi.corra@libero.it GitHub: CorraPiano
- Michele Fassini – Mail: ___ GitHub: MicheleFassini