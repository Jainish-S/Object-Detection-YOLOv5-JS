import * as tf from "@tensorflow/tfjs";
import * as tfconv from "@tensorflow/tfjs-converter";

// const BASE_PATH = "/YOLOv5_tfjs/model.json";

import { CLASSES } from "./classes";

export interface DetectedObject {
    bbox: [number, number, number, number]; // [x, y, width, height]
    class: string;
    score: number;
}

export async function load(modelPath: string) {
    if (tf == null) {
        throw new Error(`Can't find TensorFlow.js. Please import it.`);
    }

    // create object of YOLOv5 and return it after loading

    const objectDetection = new YOLOv5(modelPath);
    await objectDetection.load();
    return objectDetection;
}

export class YOLOv5 {
    private modelPath: string;
    private model: tfconv.GraphModel;

    constructor(modelUrl?: string) {
        this.modelPath = modelUrl!; // !: this is a workaround for the fact that modelUrl is optional
        this.model = null!; //* initializing it so as to compile without error.
    }

    async load() {
        this.model = await tfconv.loadGraphModel(this.modelPath);

        const zeroTensor = tf.zeros([1, 640, 640, 3]); // This is the img size on which the model is trained

        const result = (await this.model.executeAsync(
            zeroTensor
        )) as tf.Tensor[];

        await Promise.all(result.map((t) => t.data()));
        result.map((t) => t.dispose());
        zeroTensor.dispose();
    }

    private async infer(
        img:
            | tf.Tensor3D
            | ImageData
            | HTMLImageElement
            | HTMLCanvasElement
            | HTMLVideoElement,
        maxNumBoxes: number,
        minScore: number
    ): Promise<DetectedObject[]> {
        const [modelWidth, modelHeight] = this.model.inputs[0].shape.slice(
            1,
            3
        );

        const batched = tf.tidy(() => {
            if (!(img instanceof tf.Tensor)) {
                img = tf.image.resizeBilinear(tf.browser.fromPixels(img), [
                    modelWidth,
                    modelHeight,
                ]);
            }
            return img.div(255).expandDims(0);
        }); 

        const result = (await this.model.executeAsync(batched)) as tf.Tensor[];

        const boxes = result[0].dataSync() as Float32Array;
        const scores = result[1].dataSync() as Float32Array;
        const classes = result[2].dataSync() as Float32Array;
        const valid_detections = result[3].dataSync()[0];

        batched.dispose();
        tf.dispose(result);

        const preventBackend = tf.getBackend();
        if (preventBackend === "webgl") {
            tf.setBackend("cpu");
        }

        if (preventBackend !== tf.getBackend()) {
            tf.setBackend(preventBackend);
        }

        return this.buildDetectedObjects(
            modelWidth,
            modelHeight,
            boxes,
            scores,
            classes,
            valid_detections
        );
    }

    private buildDetectedObjects(
        width: number,
        height: number,
        boxes: Float32Array,
        scores: Float32Array,
        classes: Float32Array,
        valid_detections: number
    ): DetectedObject[] {
        const objects: DetectedObject[] = [];

        for (let i = 0; i < valid_detections; i++) {
            const bbox: number[] = [];
            for (let j = 0; j < 4; j++) {
                bbox[j] = boxes[i * 4 + j];
            }

            const minX = bbox[0] * width;
            const minY = bbox[1] * height;
            const maxX = bbox[2] * width;
            const maxY = bbox[3] * height;

            bbox[0] = minX;
            bbox[1] = minY;
            bbox[2] = maxX - minX;
            bbox[3] = maxY - minY;

            objects.push({
                bbox: bbox as [number, number, number, number],
                class: CLASSES[classes[i]].name,
                score: scores[i],
            });
        }

        return objects;
    }

    async detect(
        img:
            | tf.Tensor3D
            | ImageData
            | HTMLImageElement
            | HTMLCanvasElement
            | HTMLVideoElement,
        maxNumBoxes: number = 20,
        minScore: number = 0.5
    ): Promise<DetectedObject[]> {
        return this.infer(img, maxNumBoxes, minScore);
    }

    dispose() {
        if (this.model != null) {
            this.model.dispose();
        }
    }
}
