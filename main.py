from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def collate_fn(x):
    return x[0]


def face_match(img_path):  # img_path= location of photo, data_path= location of data.pt
    candidate_img = Image.open(img_path)
    candidate_face, candidate_face_probability = mtcnn(candidate_img, return_prob=True)
    candidate_image_embedding = resnet(candidate_face.unsqueeze(0)).detach()
    saved_data = torch.load('data.pt')  # loading data.pt file
    embeddings_list_from_data = saved_data[0]  # getting embedding data
    names_list_from_data = saved_data[1]  # getting list of names
    distances_list = []  # list of matched distances, minimum distance is used to identify the person

    for i, embedding_from_data in enumerate(embeddings_list_from_data):
        distance = torch.dist(candidate_image_embedding, embedding_from_data).item()
        distances_list.append(distance)
    idx_min = distances_list.index(min(distances_list))

    return names_list_from_data[idx_min], min(distances_list)


writer = SummaryWriter()

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # initializing resnet

dataset = datasets.ImageFolder('photos')  # photos folder path
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names

loader = DataLoader(dataset, collate_fn=collate_fn)
faces_list = []  # list of cropped faces from photos folder
names_list = []  # list of names corresponding to cropped photos
embeddings_list = []  # list of embedding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, face_detect_probability = mtcnn(img, return_prob=True)
    if face is not None and face_detect_probability > 0.90:  # if face detected and probability > 90%
        faces_list.append(face)
        embedding = resnet(face.unsqueeze(0)).detach()  # passing cropped face into resnet model to get embedding matrix
        writer.add_embedding(embedding, global_step=idx)
        embeddings_list.append(embedding)  # resulted embedding matrix is stored in a list
        names_list.append(idx_to_class[idx])  # names are stored in a list

data = [embeddings_list, names_list]
torch.save(data, 'data.pt')  # saving data.pt file

result = face_match('1.png')
print('Face matched with: ', result[0], 'With distance: ', result[1])
