import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from PIL import Image, ImageDraw


def load_models():
	global device, mtcnn, resnet
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	mtcnn = MTCNN(keep_all=True, device=device)
	resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
	return mtcnn, resnet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn, resnet = load_models()


def predict_image(img: str):
	img = Image.open(img)
	boxes, probs, points = mtcnn.detect(img, landmarks=True)
	return boxes, probs, points


def show_image(boxes, points, img):
	img = Image.open(img)
	img_draw = img.copy()
	draw = ImageDraw.Draw(img_draw)
	for i, (box, point) in enumerate(zip(boxes, points)):
		draw.rectangle(box.tolist(), width=5)
		for p in point:
			draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
	plt.imshow(img_draw)
	print(f"There are {len(points)} faces in the photo")


def save_piece_wise(boxes, points, img):
	img = Image.open(img)
	img_draw = img.copy()
	draw = ImageDraw.Draw(img_draw)
	if not os.path.exists("detected_faces"):
		os.makedirs("detected_faces")
	for i, (box, point) in enumerate(zip(boxes, points)):
		draw.rectangle(box.tolist(), width=5)
		for p in point:
			draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
		face_img = img.crop(box.tolist())
		face_img.save(f"detected_faces/face_{i}.jpg")


def calculate_dummies(image, boxes):
	aligned = []
	img = Image.open(image)

	boxes = boxes.astype(np.float32)
	boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)

	for j, box in enumerate(boxes_tensor):
		aligned_face = extract_face(img, box)
		aligned.append(aligned_face)

	aligned_tensor = torch.stack(aligned).to(device)

	embeddings = resnet(aligned_tensor).detach().cpu()
	dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
	return dists


def calculate_face_embedding(imagePath: str):
	image = Image.open(imagePath)
	face = mtcnn(image)
	if face is not None:
		embedding = resnet(face)
		return embedding
	else:
		return None


def return_classes(directory: str) -> dict:
	image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
	names = [i.split(".png")[0] for i in image_files]
	name_to_class = {}
	for i in range(len(names)):
		name_to_class[names[i]] = image_files[i]

	return name_to_class


def process_reference_images(root_dir: str, classes: dict):
	images = []
	for i in classes.values():
		try:
			images.append(
				calculate_face_embedding(f'{root_dir}/{i}')
			)
		except Exception as exception:
			print(f"Error in processing {root_dir}/{i}", exception)
	return images


def calculate_similar(imageVal: torch.tensor, overall: torch.tensor, itrn_id: int, classes, root_dir, debug: bool = False):
	minVal = float('inf')
	argMin = 0
	for i in range(overall.shape[0]):
		val = (imageVal - overall[i]).norm()
		if val < minVal:
			minVal = val
			argMin = i
			if debug:
				print(f"The local minimum norm for face_{i} is {minVal}")

	if minVal < 0.76:
		if debug:
			print(f"The global minimum norm for face_{argMin} is {minVal}")
			print(f"Identified face: {list(classes.keys())[itrn_id]}")
		globalMin = f'./detected_faces/face_{argMin}.jpg'

		return globalMin, f"{root_dir}/{list(classes.keys())[itrn_id]}.png"

	else:
		if debug:
			print("No global minimum under filter")
		return None


def run_all(imgStr: str, root_dir: str):
	boxes, probs, points = predict_image(imgStr)
	# show_image(boxes, points, imgStr)

	save_piece_wise(boxes, points, imgStr)

	classes = return_classes(root_dir)
	references = process_reference_images(root_dir, classes)
	overall = calculate_face_embedding(imgStr)

	images = []
	for i in range(len(references)):
		# print(f"Iteration {i}")
		image = calculate_similar(references[i], overall, i, classes, root_dir, debug=False)
		if image:
			images.append(image)
	show_image = True
	if show_image:
		fig, axarr = plt.subplots(len(images), 2, figsize=(20, 50))
		plt.tight_layout()
		plt.axis('off')
		i = 0
		for image in images:
			try:
				axarr[i, 0].imshow(Image.open(image[0]))
				axarr[i, 1].imshow(Image.open(image[1]))
				axarr[i, 0].axis('off')
				axarr[i, 1].axis('off')
				axarr[i, 0].set_title(image[0])
				axarr[i, 1].set_title(image[1])
				i += 1
			except:
				pass

		plt.savefig('output.png')



if __name__ == "__main__":
	imgStr = '../test_images/outdoors.jpg'
	root_dir = '../standard_set'
	run_all(imgStr, root_dir)
