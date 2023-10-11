import cv2
import face_recognition as fr

imgLatrel = fr.load_image_file('latrel.jpg')
imgLatrel = cv2.cvtColor(imgLatrel, cv2.COLOR_BGR2RGB)
imgLatrelTeste = fr.load_image_file('latrel_teste.jpg')
imgLatrelTeste = cv2.cvtColor(imgLatrelTeste, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgLatrel)[0]
cv2.rectangle(imgLatrel, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

encodeLatrel = fr.face_encodings(imgLatrel)[0]
encodeLatrelTeste = fr.face_encodings(imgLatrelTeste)[0]

comparacao = fr.compare_faces([encodeLatrel], encodeLatrelTeste)
distancia = fr.face_distance([encodeLatrel], encodeLatrelTeste)

print(comparacao,distancia)
cv2.imshow('Latrel', imgLatrel)
cv2.imshow('Latrel Teste', imgLatrelTeste)
cv2.waitKey(0)

