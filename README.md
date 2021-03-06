# DALL-E Encoder/Decoder for JavaScript

A JavaScript port of OpenAI's DALL-E encoder/decoder (using TensorFlow.js). Works in the browser but takes about 200x longer to encode/decode compared to Pytorch (not sure why - I'd have expected an order of magnitude difference but not >2 OOM). Should work in Deno once it gets `OffscreenCanvas` and WebGL support.

# Demo

https://josephrocca.github.io/dalle-encoder-decoder-js/demo.html

# Usage

```js
import {encode, decode} from "./mod.js";

let latent = await encode("https://i.imgur.com/2BU6QVe.jpg"); // the encode function accepts a URL, or Image, or ImageData, or ImageBitmap, or Canvas, or OffscreenCanvas, etc.
console.log(latent);
let pixelData = await decode(latent);
console.log(pixelData);

// see demo.html for an expanded example
```
