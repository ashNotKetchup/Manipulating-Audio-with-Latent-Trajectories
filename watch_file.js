const fs = require('fs');
const path = require('path');
const maxApi = require("max-api");


// The relative path to the file you want to watch
// const relativePath = './audio/output.wav';

// Resolve the absolute (full) path
// const fullPath = path.resolve(relativePath);
const fullPath = "/Users/ash/Documents/GitHub/Timbre-Slider/audio/output.wav";

maxApi.outlet(fullPath);

console.log(`Watching file: ${fullPath}`);
maxApi.outlet(fullPath);
fs.watch(fullPath, (eventType, filename) => {
  if (filename) {
    console.log(`File changed: ${fullPath}`);
    maxApi.outlet(`${fullPath}`);
  } else {
    // Some platforms don’t provide filename, so fallback to fullPath
    console.log(`Change detected in: ${fullPath}`);
    maxApi.outlet(`${fullPath}`);
  }
});

