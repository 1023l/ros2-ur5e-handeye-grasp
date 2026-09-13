# YOLO weights (not committed — download locally)

```bash
pip3 install --user ultralytics
ros2 run arm_system setup_yolo_weights.py
```

Produces:
- `yolov8n.pt` — YOLOv8 nano COCO weights
- `demo_target.jpg` — crop used in synthetic YOLO scenes
