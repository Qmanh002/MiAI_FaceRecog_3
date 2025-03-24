from __future__ import absolute_import, division, print_function
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
from collections import Counter
import traceback
from image_saver import ImageSaver

def load_models(face_classifier_path, mask_classifier_path):
    """Load classification models with validation"""
    models = {}
    try:
        with open(face_classifier_path, 'rb') as f:
            models['face_model'], models['face_names'] = pickle.load(f)
        with open(mask_classifier_path, 'rb') as f:
            models['mask_model'], models['mask_names'] = pickle.load(f)
        print("Models loaded successfully")
        return models
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        print(traceback.format_exc())
        raise

def initialize_facenet(model_path):
    """Initialize FaceNet with validation"""
    try:
        print('Loading feature extraction model')
        facenet.load_model(model_path)
        graph = tf.get_default_graph()
        return (
            graph.get_tensor_by_name("input:0"),
            graph.get_tensor_by_name("embeddings:0"),
            graph.get_tensor_by_name("phase_train:0")
        )
    except Exception as e:
        print(f"FaceNet initialization failed: {str(e)}")
        raise

def validate_face_region(frame, x1, y1, x2, y2):
    """Comprehensive face region validation"""
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    if any(not isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
        return False
    if x1 >= x2 or y1 >= y2:
        return False
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return False
    if (y2 - y1) / frame.shape[0] < 0.1:
        return False
    return True

def safe_crop(frame, x1, y1, x2, y2):
    """Safe cropping with boundary checks"""
    try:
        cropped = frame[max(0, y1):min(frame.shape[0], y2), 
                       max(0, x1):min(frame.shape[1], x2)]
        if cropped.size == 0:
            raise ValueError("Empty crop result")
        return cropped
    except Exception as e:
        print(f"Cropping failed: {str(e)}")
        raise

def process_frame(frame, boxes, facenet_tensors, models, thresholds, image_saver):
    """Process all faces in a single frame"""
    processed = []
    for box in boxes:
        try:
            x1, y1, x2, y2 = list(map(int, box[:4]))
            if not validate_face_region(frame, x1, y1, x2, y2):
                continue

            # Preprocessing
            cropped = safe_crop(frame, x1, y1, x2, y2)
            scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
            scaled = facenet.prewhiten(scaled)
            face_image = scaled.reshape(-1, 160, 160, 3)

            # Feature extraction
            feed_dict = {
                facenet_tensors[0]: face_image,
                facenet_tensors[2]: False
            }
            emb_array = facenet_tensors[1].eval(
                feed_dict=feed_dict,
                session=tf.get_default_session()
            )

            # Mask detection
            mask_pred = models['mask_model'].predict(emb_array)[0]
            mask_prob = models['mask_model'].predict_proba(emb_array)[0][mask_pred]
            mask_status = models['mask_names'][mask_pred] if mask_prob >= thresholds['mask'] else "Unknown"

            # Prepare results
            result = {
                'box': (x1, y1, x2, y2),
                'mask_status': mask_status,
                'mask_prob': mask_prob,
                'color': (0, 255, 0)  # Default green (mask)
            }

            # Face recognition for no-mask cases
            if mask_status == "NoMask":
                result['color'] = (0, 0, 255)  # Red
                face_pred = models['face_model'].predict(emb_array)[0]
                face_prob = models['face_model'].predict_proba(emb_array)[0][face_pred]
                
                if face_prob >= thresholds['face']:
                    result['face_name'] = models['face_names'][face_pred]
                    result['face_prob'] = face_prob
                    # Save the face image using ImageSaver
                    image_saver.save_face_image(frame, (x1, y1, x2, y2), result['face_name'])
                else:
                    result['face_name'] = "Unknown"

            processed.append(result)

        except Exception as e:
            print(f"Face processing skipped: {str(e)}")
            continue

    return processed

def draw_results(frame, results, image_saver):
    """Draw all detection results on frame"""
    for res in results:
        x1, y1, x2, y2 = res['box']
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), res['color'], 2)
        
        # Draw mask info
        text_y = y2 + 20
        mask_text = f"{res['mask_status']}: {res['mask_prob']:.2f}"
        cv2.putText(frame, mask_text, (x1, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw face info if available
        if res['mask_status'] == "NoMask":
            face_text = res.get('face_name', 'Unknown')
            if 'face_prob' in res:
                face_text += f": {res['face_prob']:.2f}"
            cv2.putText(frame, face_text, (x1, text_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add status message showing last capture time
    status = f"Last capture: {image_saver.get_last_capture_time()}"
    cv2.putText(frame, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def camera_loop(cap, pnet, rnet, onet, facenet_tensors, models, config):
    """Main camera processing loop"""
    last_valid_time = time.time()
    image_saver = ImageSaver()
    
    while True:
        try:
            # Frame reading with timeout
            frame = cap.read()
            if frame is None:
                if time.time() - last_valid_time > 5:  # 5 second timeout
                    raise RuntimeError("Camera feed timeout")
                continue
            
            last_valid_time = time.time()
            frame = imutils.resize(frame, width=600)
            frame = cv2.flip(frame, 1)

            # Face detection
            boxes, _ = align.detect_face.detect_face(
                frame, config['minsize'], pnet, rnet, onet,
                config['threshold'], config['factor'])

            # Process frame
            results = process_frame(frame, boxes, facenet_tensors, models, {
                'face': config['face_thresh'],
                'mask': config['mask_thresh']
            }, image_saver)

            # Display results
            draw_results(frame, results, image_saver)
            cv2.imshow('Face Mask Detection', frame)

            # Exit check
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            print(traceback.format_exc())
            break

def main():
    # Configuration
    config = {
        'minsize': 20,
        'threshold': [0.6, 0.7, 0.7],
        'factor': 0.709,
        'face_thresh': 0.4,
        'mask_thresh': 0.1,
        'min_face_size': 0.1
    }

    # Initialize system
    cap = None
    try:
        # Load models
        models = load_models('Models/facemodel.pkl', 'Models/maskmodel.pkl')
        
        # Initialize FaceNet
        facenet_tensors = initialize_facenet('Models/20180402-114759.pb')
        
        # Initialize MTCNN
        with tf.Session() as sess:
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")
            
            # Start camera with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    cap = VideoStream(src=0).start()
                    time.sleep(2.0)  # Warmup
                    if cap.stream.isOpened():
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
            
            # Main processing loop
            camera_loop(cap, pnet, rnet, onet, facenet_tensors, models, config)

    except Exception as e:
        print(f"System initialization failed: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        if cap is not None:
            cap.stop()
        cv2.destroyAllWindows()
        # Explicit TensorFlow session cleanup
        tf.reset_default_graph()

if __name__ == "__main__":
    main()