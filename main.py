# -*- coding: utf-8 -*-
import cv2
import dlib  # 需先pip install CMake
'''
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 調整預設影像大小(預設值很大，很吃效能)
camera.set(cv2. CAP_PROP_FRAME_WIDTH, 650)
camera.set(cv2. CAP_PROP_FRAME_HEIGHT, 500)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while(camera.isOpened()):
    # 讀出frame資訊
    ret, frame = camera.read()

    # 偵測人臉
    face_rects, scores, idx = detector.run(frame, 0)
    # face_rects 偵測臉上下左右(方框)的位置，為兩個座標點 (x1, y1)、(x2, y2) => list
    # scores 為偵測人臉的分數，分數越高判斷人臉的機率越大 => list
    # idx 為判斷人臉的方向{0:正臉, 1:左側臉, 2:右側臉, 3:右歪臉, 4:左歪臉} => list

    # 取出偵測的結果
    for i, d in enumerate(face_rects):  # 因為 face_rects 的 list 內只有一個 index，故 i==0
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = " %2.2f ( %d )" % (scores[i], idx[i])

        # 繪製出偵測人臉的矩形範圍
        # cv2.rectangle(照片, 起始點, 結束點, 框線的顏色, 框線的粗細, 框線樣式)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        # 標上人臉偵測分數與人臉方向子偵測器編號
        # cv2.putText(影像, 文字, 左下座標, 字型, 大小(縮放比例), 顏色, 線條寬度, 線條種類)
        # FONT_HERSHEY_DUPLEX：正常大小無襯線字體
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,  0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # 給68特徵點辨識取得一個轉換顏色的frame
        landmarks_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 找出特徵點位置
        # shape 內紀錄了所有特徵點的座標，68個(x,y)
        shape = predictor(landmarks_frame, d)

        # 繪製68個特徵點
        for j in range(68):
            cv2.circle(frame, (shape.part(j).x, shape.part(j).y), 3, (0, 0, 255), 2)
            cv2.putText(frame, str(j), (shape.part(j).x, shape.part(j).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    # 輸出到畫面
    cv2.imshow("Face Detection", frame)

    # 如果按下ESC键，就退出
    # ESC 的 ASCII 碼為 27
    if cv2.waitKey(10) == 27:
        break

# 釋放記憶體
camera.release()
# 關閉所有視窗
cv2.destroyAllWindows()
'''

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 這裡的 file 要寫絕對位置，不能寫相對位置!!!
# cv_face_detect = cv2.CascadeClassifier('C:\\Users\\Case110208\\Desktop\\face_detect_ai\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')

img = cv2.imread('bts.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_rects, scores, idx = detector.run(img, 0)

if scores:
    for i, d in enumerate(face_rects):  # 因為 face_rects 的 list 內只有一個 index，故 i==0
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = " %2.2f ( %d )" % (scores[i], idx[i])

        # 繪製出偵測人臉的矩形範圍
        # cv2.rectangle(照片, 起始點, 結束點, 框線的顏色, 框線的粗細, 框線樣式)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        # 標上人臉偵測分數與人臉方向子偵測器編號
        # cv2.putText(影像, 文字, 左下座標, 字型, 大小(縮放比例), 顏色, 線條寬度, 線條種類)
        # FONT_HERSHEY_DUPLEX：正常大小無襯線字體
        cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,  0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # 給68特徵點辨識取得一個轉換顏色的frame
        landmarks_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 找出特徵點位置
        # shape 內紀錄了所有特徵點的座標，68個(x,y)
        shape = predictor(landmarks_frame, d)

        # 繪製68個特徵點
        # for j in range(68):
        #     cv2.circle(img, (shape.part(j).x, shape.part(j).y), 3, (0, 0, 255), 2)
        #     cv2.putText(img, str(j), (shape.part(j).x, shape.part(j).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    # 顯示成果
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # 正常視窗大小
    # cv2.imshow('img', img)                     # 秀出圖片
    # cv2.imwrite("result.jpg", img)             # 保存圖片
    # cv2.waitKey(0)                             # 等待按下任一按鍵
    # cv2.destroyAllWindows()                    # 關閉視窗
    print("圖片上傳成功")
else:
    print("這不是一張含有人臉的相片!!!")

# # 偵測臉部
# faces = cv_face_detect.detectMultiScale(
#     gray,
#     scaleFactor=1.08,
#     minNeighbors=5,
#     # minSize=(96, 96)
# )
#
# # 繪製人臉部份的方框
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)



