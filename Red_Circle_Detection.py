import cv2
import numpy as np
#Gerekli kütüphaneler yüklenir
cap = cv2.VideoCapture(0)
#0 İD'li kamera "cap" olarak adlandırılır
while True:
    ret,frame = cap.read()
    #Kameradan okunan görüntüler "frame" değişkenine atılır

    if ret==False:
        break
    #Programın sonlanması dahilinde algoritmaya feedback yapılır.

    frame = cv2.flip(frame, 1)
    #Alınan görüntü Y eksenine göre yansıtılır. (Sağ el kaldırıldığında görüntüde de sağ el kalkar)

    frame_lab = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
    #Alınan görüntü BGRA formatından BGR formatına dönüştürülür

    frame_lab = cv2.medianBlur(frame_lab, 3)
    #Görüntü üzerinde BLURLAMA yapılır

    frame_lab =  cv2.cvtColor(frame_lab, cv2.COLOR_BGR2Lab)
    #Görüntü BGR2Lab formatına dönüştürülür

    lower_red = np.array([20, 150, 150])
    #Algılamak istediğimiz en düşük KIRMIZI renk kodu belirlenir

    upper_red = np.array([190, 255, 255])
    #Algılamak istediğimiz en yüksek KIRMIZI renk kodu belirlenir

    masked_frame = cv2.inRange(frame_lab,lower_red, upper_red)
    #İstediğimiz Kırmızı renk skalasındaki alan "masked_frame" değişkenine atılır.

    masked_frame = cv2.GaussianBlur(masked_frame, (5, 5), 2, 2)
    #İstediğimiz Kırmızı renk skalasındaki görüntü Blurlanır

    circles = cv2.HoughCircles(masked_frame, cv2.HOUGH_GRADIENT, 1, masked_frame.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=60)
    #Hough Circle fonksiyonu görüntü üzerine hayali çemberler çizerek gerçek daire tespiti yapmamızı sağlar
    #Tespit ettiği daireleri "circles" içine atar
    #Parametreler : İşlenecek görüntü , Çember tespiti için kullanılacak method,Ölçek değeri ,Çemberler arası mesafe,ilk eşik değeri,ikinci eşik değeri,
    #Görüntü üzerindeki çemberlerin minimum çapı,görüntü üzerindeki çemberlerin maksimum çapı

    #Eğer circles boş değilse
    if circles is not None:
        #Veri tipi dönüşümü yapılır
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle merkez koordinatları

            cv2.circle(frame, center, 1, (0, 255, 0), 3)
            # Merkeze küçük circle çizilir

            radius = i[2] #Yarıcap
            cv2.circle(frame, center, radius, (0, 255, 0), 3)
            #Yarıcaplı bir circle çizilir

            cv2.putText(frame, "Merkez", (center), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            #Merkez noktasına "Merkez" çıktısı yazılır

    cv2.imshow("Mask",masked_frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()