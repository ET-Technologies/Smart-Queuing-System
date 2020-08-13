
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
from os import path

class Queue:
    '''
    Class for dealing with queues
    '''

    def __init__(self):
        self.queues = []

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.model = None

        try:
            self.model = IENetwork(self.model_structure, self.model_weights) # old version
            #self.model = IECore.read_network(self.model_structure, self.model_weights) # new version but does not work
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        print ("--------")    
        print ("Model is loaded as self.model: " + str(self.model))
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.exec_network = None
        
        print("input_name: " + str(self.input_name))
        print("input_shape: " + str(self.input_shape))
        print("output_name: " + str(self.output_name))
        print("output_shape: " + str(self.output_shape))
        
        print ("--------")

    def load_model(self):
        # This function loads the model to the network
        self.core = IECore()
        self.exec_network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        print ("Model is loaded to the network as self.exec_network")
        print ("--------")

    def predict(self, image, initial_w, initial_h):
        # This function returns the coordinats of the bounding boxes and the image
        print ("--")
        print ("Start predictions")
        requestid = 0
        preprocessed_image = self.preprocess_input(image)
        # Starts synchronous inference
        outputs = self.exec_network.infer({self.input_name: preprocessed_image})
        print ("Output of the inference request: " + str(outputs))
        outputs = self.exec_network.requests[requestid].outputs[self.output_name]
        coords, image = self.boundingbox(outputs, image, initial_w, initial_h)
        
        print ("End predictions")
        print ("--------")
        return coords, image
    
    def preprocess_input(self, image):
        # In this function the original image is resized, transposed and reshaped to fit the model requirements.
        print ("--------")
        print ("Start: preprocess image")
        self.width = int(image.shape[1])
        self.height = int(image.shape[0])
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        print ("Original image size is (W x H): " + str(self.width) + "x"+ str(self.height))
        print ("Image is now [BxCxHxW]: " + str(image.shape))
        print ("End: preprocess image")
        print ("--------")
                        
        return image
    
    def boundingbox(self, outputs, image, initial_w, initial_h):
        coords = []
        coords_x_y =[]
        print ("--------")
        print ("Start: boundingbox")
        print ("Bounding box input: " + str(outputs))
        print ("Coords: " + str(coords))
        print ("Original image size is (W x H): " + str(initial_w) + "x"+ str(initial_h))
        for obj in outputs[0][0]:
            # Draw bounding box for object when it's probability is more than the specified threshold
            if obj[2] > self.threshold:
                #variant 1
                obj[3] = int(obj[3] * initial_w)
                obj[4] = int(obj[4] * initial_h)
                obj[5] = int(obj[5] * initial_w)
                obj[6] = int(obj[6] * initial_h)
                #cv2.rectangle(image, (obj[3], obj[4]), (obj[5], obj[6]), (0, 55, 255), 1)
                coords.append([obj[3],obj[4],obj[5],obj[6]])
                
                #variant 2
                xmin = int(obj[3])
                ymin = int(obj[4])
                xmax = int(obj[5])
                ymax = int(obj[6])
                coords_x_y.append([xmin,ymin,xmax,ymax])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 10,4)
                
                print ("Coords: " + str(coords))
                print ("Coords coords_x_y ...: " + str(coords_x_y))
        print ("Coords in total: " + str(coords))
        print ("Coords in total coords_x_y: " + str(coords_x_y))
        print ("End: boundingbox")
        print ("--------")
        
        return coords_x_y, image    

def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path

    start_model_load_time = time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time
    

    queue = Queue()

    try:
        queue_param = np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        print("Reading video file", video_file)
        cap = cv2.VideoCapture(video_file)
        cap.open(video_file)
        if not path.exists(video_file):
            print("Cannot locate video file: " + video_file)
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (initial_w, initial_h), True)

    counter = 0
    start_inference_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1

            coords, image = pd.predict(frame, initial_w, initial_h)
            num_people = queue.check_coords(coords)
            print("Total People in frame =", len(coords))
            print("Number of people in queue =", num_people)
            out_text = ""
            y_pixel = 25

            for k, v in num_people.items():
                out_text += "No. of People in Queue " + str(k) + " is " + str(v)
                if v >= int(max_people):
                    out_text += " Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text = ""
                y_pixel += 40
            out_video.write(image)

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time) + '\n')
            f.write(str(fps) + '\n')
            f.write(str(total_model_load_time) + '\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    args = parser.parse_args()

    main(args)