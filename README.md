# DALL-E Encoder/Decoder for JavaScript

A JavaScript port of OpenAI's DALL-E encoder/decoder (using TensorFlow.js). Works in the browser but takes about 200x longer to encode/decode compared to Pytorch.

# Demo

https://josephrocca.github.io/dalle-encoder-decoder-js/demo.html

# Usage

```html
<script type=module>
  import {encode, decode} from "./mod.js";

  let latent = await encode("https://i.imgur.com/2BU6QVe.jpg");
  console.log(latent);
  let pixelData = await decode(latent);
  console.log(pixelData);
  
  // see demo.html for an expanded example
</script>
```
