from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
import argparse
import facenet
import imutils
import numpy as np
import cv2
import time
import pickle
import align.detect_face

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    # Cấu hình tham số
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]  # Ngưỡng phát hiện khuôn mặt
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160
    MASK_CLASSIFIER_PATH = 'Models/maskmodel.pkl'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
    CONFIDENCE_THRESHOLD = 0.7  # Ngưỡng tin cậy

    # Load model phát hiện khẩu trang
    try:
        with open(MASK_CLASSIFIER_PATH, 'rb') as file:
            mask_model, class_names = pickle.load(file)
        print("Mask Classifier loaded successfully")
    except Exception as e:
        print(f"Error loading mask model: {str(e)}")
        return

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

        with sess.as_default():
            # Load FaceNet model để trích xuất đặc trưng
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lấy các tensor input/output
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Khởi tạo MTCNN để phát hiện khuôn mặt
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            cap = VideoStream(src=0).start()
            time.sleep(2.0)  # Chờ camera khởi động

            while True:
                frame = cap.read()
                if frame is None:
                    continue
                    
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                # Phát hiện khuôn mặt
                bounding_boxes, _ = align.detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        
                        # Kiểm tra kích thước khuôn mặt
                        face_height = bb[i][3] - bb[i][1]
                        if face_height / frame.shape[0] < 0.15:  # Bỏ qua khuôn mặt quá nhỏ
                            continue
                            
                        try:
                            # Cắt và tiền xử lý ảnh khuôn mặt
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                              interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            
                            # Trích xuất đặc trưng
                            feed_dict = {images_placeholder: scaled_reshape, 
                                       phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)

                            # Phát hiện khẩu trang
                            predictions = mask_model.predict_proba(emb_array)
                            best_class_idx = np.argmax(predictions, axis=1)
                            best_prob = predictions[0][best_class_idx[0]]
                            label = class_names[best_class_idx[0]]
                            
                            # Xác định màu sắc và nhãn hiển thị
                            if best_prob >= CONFIDENCE_THRESHOLD:
                                if label == "Mask":
                                    color = (0, 255, 0)  # Xanh lá - Đeo khẩu trang
                                    text = f"{label}: {best_prob:.2f}"
                                else:
                                    color = (0, 0, 255)  # Đỏ - Không đeo khẩu trang
                                    text = f"No Mask: {best_prob:.2f}"
                            else:
                                color = (255, 255, 0)  # Vàng - Không chắc chắn
                                text = "Unknown"
                            
                            # Vẽ khung và hiển thị thông tin
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), 
                                        (bb[i][2], bb[i][3]), color, 2)
                            cv2.putText(frame, text, 
                                      (bb[i][0], bb[i][3] + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
                        except Exception as e:
                            print(f"Error processing face: {str(e)}")
                            continue

                # Hiển thị tỉ lệ FPS
                fps = cap.stream.get(cv2.CAP_PROP_FPS)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('Mask Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()