from ultralytics import YOLO

# Load a COCO-pretrained model
model = YOLO("yolov8s-seg.pt")

results = model("NY.jpg")

for result in results:
    result.show()  # display to screen
