import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.4.0/dist/tf.fesm.js";

await tf.setBackend("webgl");

let encoderModel = await tf.loadGraphModel(`./models/encoder/model.json`);
let decoderModel = await tf.loadGraphModel(`./models/decoder/model.json`);

const logitLaplaceEps = 0.1;

export async function encode(img, opts={}) {

  if(typeof img === "string" && (img.startsWith("https://") || img.startsWith("http://"))) {
    let blob = await fetch(img, {referrer:""}).then(r => r.blob()); // {referrer:""} is just because imgur doesn't blocks 127.0.0.1 referrer which is annoying during testing
    img = await createImageBitmap(blob);
  }

  let canvas = new OffscreenCanvas(256, 256);
  let ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let rgbData = [[], [], []]; // [r, g, b]
  // remove alpha and put into correct shape:
  let d = imageData.data;
  for(let i = 0; i < d.length; i += 4) {
    let x = (i/4) % canvas.width;
    let y = Math.floor((i/4) / canvas.width)
    if(!rgbData[0][y]) rgbData[0][y] = [];
    if(!rgbData[1][y]) rgbData[1][y] = [];
    if(!rgbData[2][y]) rgbData[2][y] = [];
    rgbData[0][y][x] = (1 - 2*logitLaplaceEps) * (d[i+0]/255) + logitLaplaceEps; // r
    rgbData[1][y][x] = (1 - 2*logitLaplaceEps) * (d[i+1]/255) + logitLaplaceEps; // g
    rgbData[2][y][x] = (1 - 2*logitLaplaceEps) * (d[i+2]/255) + logitLaplaceEps; // b
  }

  let input = {'unknown': tf.tensor([rgbData], [1,3,256,256], "float32")};
  let output = encoderModel.execute(input, ["Identity"]);
  output = output.argMax(1);
  let outputArray = (await output.array())[0];
  await decode(outputArray);
  return outputArray;
};

export async function decode(input, opts={}) {
  
  input = tf.oneHot(input, 8192)
    .expandDims(0)
    .transpose([0, 3, 1, 2]);

  input = tf.cast(input, 'float32');
  
  input = {'unknown': input};
  let output = decoderModel.execute(input, ["Identity"]);
  
  output = output.sigmoid()
    .slice([0, 0, 0, 0], [1, 3, 256, 256])
    .transpose([0, 2, 3, 1]).squeeze()
    .sub(logitLaplaceEps)
    .div(1 - 2*logitLaplaceEps)
    .maximum(0)
    .minimum(1);

  return await tf.browser.toPixels(output);
};
