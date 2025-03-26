# MiAI_FaceRecog_3
Nhận diện khuôn mặt khá chuẩn xác bằng MTCNN và Facenet!
chạy trong môi trường Python 3.7 
tải Anaconda > Environ...> Create > py 3.7
vào vs code > termial > code file .py > chọn môi trường góc phải > click chuột phải Run python 

python src/align_dataset_mtcnn.py  Dataset/DataMask/raw Dataset/DataMask/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

python src/train_mask.py TRAIN Dataset/DataMask/processed Models/20180402-114759.pb Models/maskmodel.pkl --batch_size 1000

xử lí tiền dữ liệu: 
python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25
trích xuất đặc trưng và train:
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000

file main:
python src/face_rec_cam.py 

cài môi trường:
pip install -r requirements.txt

