#!/usr/bin/env python3
import cv2
import depthai as dai

# Create pipeline
device = dai.Device()
with dai.Pipeline(device) as pipeline:
    outputQueues = {}
    sockets = device.getConnectedCameras()
    for socket in sockets:
        cam = pipeline.create(dai.node.Camera).build(socket)
        # Request smaller output instead of full resolution
        outputQueues[str(socket)] = cam.requestOutput(
            (640, 480),                # smaller resolution
            dai.ImgFrame.Type.BGR888p,
            dai.ImgResizeMode.LETTERBOX,
            30
        ).createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        for name, queue in outputQueues.items():
            videoIn = queue.get()
            assert isinstance(videoIn, dai.ImgFrame)
            cv2.imshow(name, videoIn.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
