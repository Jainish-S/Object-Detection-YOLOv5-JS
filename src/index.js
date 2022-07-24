import React, { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import "./styles.css";

import * as YOLOv5 from "./YOLOv5";

const BASE_PATH = "/YOLOv5_tfjs/model.json";

class App extends React.Component {
    videoRef = React.createRef();
    canvasRef = React.createRef();

    componentDidMount() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            const webCamPromise = navigator.mediaDevices
                .getUserMedia({
                    audio: false,
                    video: {
                        facingMode: "user",
                    },
                })
                .then((stream) => {
                    window.stream = stream;
                    this.videoRef.current.srcObject = stream;
                    return new Promise((resolve, reject) => {
                        this.videoRef.current.onloadedmetadata = () => {
                            resolve();
                        };
                    });
                });

            const modelPromise = YOLOv5.load(BASE_PATH);

            Promise.all([modelPromise, webCamPromise]).then((values) => {
                if (!values) {
                    console.log("Error");
                } else {
                    this.detectFrame(this.videoRef.current, values[0]);
                }
            });
        }
    }

    detectFrame = (video, model) => {
        model.detect(video).then((predictions) => {
            this.renderPredictions(predictions);
            requestAnimationFrame(() => {
                this.detectFrame(video, model);
            });
        });
    };

    renderPredictions = (predictions) => {
        const ctx = this.canvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        const font = "16px sans-serif";
        ctx.font = font;
        ctx.textBaseline = "top";

        predictions.forEach((prediction) => {
            const x = prediction.bbox[0];
            const y = prediction.bbox[1];
            const width = prediction.bbox[2];
            const height = prediction.bbox[3];

            ctx.strokeStyle = "#00FFFF";
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);

            ctx.fillStyle = "#00FFFF";
            const textWidth = ctx.measureText(
                prediction.class + " " + prediction.score.toFixed(2)
            ).width;
            const textHeight = parseInt(font, 10);

            ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
        });

        predictions.forEach((prediction) => {
            const x = prediction.bbox[0];
            const y = prediction.bbox[1];
            const text = prediction.class + " " + prediction.score.toFixed(2);

            ctx.fillStyle = "#000000";
            ctx.fillText(text, x, y);
        });
    };

    render() {
        return (
            <>
                <video
                    className="size"
                    autoPlay
                    playsInline
                    muted
                    ref={this.videoRef}
                    width="640"
                    height="480"
                />
                <canvas
                    className="size"
                    ref={this.canvasRef}
                    width="640"
                    height="480"
                />
            </>
        );
    }
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
    <StrictMode>
        <App />
    </StrictMode>
);
