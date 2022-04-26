import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path="v5l.pt")
labels = ["mask weared incorrect", "with mask", "without mask"]
frame = cv2.imread("face.jpg")

frame = [frame]
result = model(frame)
cord, label = result.xyxyn[0][:, :-1], result.xyxyn[0][:, -1]
x_shape, y_shape = frame[0].shape[1], frame[0].shape[0]
for i in range(len(label)):
    row = cord[i]
    if row[4] > 0.3:
        x1, y1, x2, y2= int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
        cv2.rectangle(frame[0], (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame[0], labels[int(label[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imwrite("detecton.jpg", frame[0])


# cap = cv2.VideoCapture(1)
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640,480))
# while True:
#     ret, frame = cap.read()
#     frame = [frame]
#     result = model(frame)
#     cord, label = result.xyxyn[0][:, :-1], result.xyxyn[0][:, -1]
#     x_shape, y_shape = frame[0].shape[1], frame[0].shape[0]
#     for i in range(len(label)):
#         row = cord[i]
#         if row[4] > 0.3:
#             x1, y1, x2, y2= int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
#             cv2.rectangle(frame[0], (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame[0], labels[int(label[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
#
#     out.write(frame[0])
#     cv2.imshow("frame", frame[0])
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()


