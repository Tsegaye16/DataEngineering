import cv2
import os
import re

def capture_face_photos(image_name, num_photos):
    label = ""  
    msg = 'Dataset generated successfully.'

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    output_folder = 'E:/DataEngineering/face_recognition/data'
  

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    camera = cv2.VideoCapture(0)
    count = 0

    while count < num_photos:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            output_path = os.path.join(output_folder, f"{image_name}_{count}.jpg")
            cv2.imwrite(output_path, face)
            count += 1

        cv2.imshow('Capture Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(msg)







def identify_user(known_faces_dir):
    known_faces = []
    known_names = []

    for filename in os.listdir(known_faces_dir):
        image = cv2.imread(os.path.join(known_faces_dir, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        known_faces.append(gray)
        known_names.append(os.path.splitext(filename)[0])

    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:  
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            found = False

            for i, known_face in enumerate(known_faces):
                similarity = cv2.compareHist(cv2.calcHist([face], [0], None, [256], [0, 256]),
                                             cv2.calcHist([known_face], [0], None, [256], [0, 256]),
                                             cv2.HISTCMP_CORREL)
                if similarity > 0.90:
                    name = known_names[i]
                    found = True
                    break

            if not found:
                name = "Unknown"
            name = re.sub(r'[\d_]', '', name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    video_capture.release()
    cv2.destroyAllWindows()



import cv2
import os
import re

def train_and_identify():
    
    name = input("Enter the name of the person to train: ")
    num_images = int(input("Enter the number of images to capture: "))

 
    print("Capturing face photos for training...")
    capture_face_photos(name, num_images)

  
    print("Identifying the user...")
    identify_user('C:/Users/Tsegaye/Documents/ML/data')


train_and_identify()