const maxApi = require("max-api");
const DICT_ID = "representations.dict";

// send.js
const dgram = require('dgram');
const socket = dgram.createSocket('udp4');

// Destination settings
const HOST = '127.0.0.1';  // or a remote IP like '192.168.1.42'
const PORT = 9997;


// Used for storing the initial value
let initialDict = {};

// Getting and setting dicts is an asynchronous process and the API function
// calls all return a Promise. We use the async/await syntax here in order
// to handle the async behaviour gracefully.
//
// Want to learn more about Promised and async/await:
//		* Web Fundamentals intro to Promises: https://developers.google.com/web/fundamentals/primers/promises
//		* Promises Deep Dive on MDN: https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous/Promises
//		* Web Fundamentals on using async/await and their benefits: https://developers.google.com/web/fundamentals/primers/async-functions
//		* Async Functions on MDN: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function

maxApi.addHandlers({
	set: async (path, value) => {
		const dict = await maxApi.updateDict(DICT_ID, path, value);
		await maxApi.outlet(dict);
	},
	receive: async (path, value) => {
		const dict = await maxApi.updateDict(DICT_ID, path, value);
		await maxApi.outlet(dict);
	},	
	reset: async () => {
		const dict = await maxApi.setDict(DICT_ID, initialDict);
		await maxApi.outlet(dict);
	},
send: async () => {
	const dict = await maxApi.getDict(DICT_ID);
	let json;
	try {
		json = JSON.stringify(dict);
	} catch (err) {
		console.error('Failed to stringify dict for sending:', err);
		json = '{}';
	}
	const prefix = 'sending_latent ';
	const message = Buffer.from(prefix + json);

	socket.send(message, 0, message.length, PORT, HOST, (err) => {
	  if (err) console.error('Send error:', err);
	  else console.log(`UDP message sent to ${HOST}:${PORT}`);
	  // keep socket open for subsequent sends
	});
	
	await maxApi.outlet(dict);
},
}
);



// We use this to store the initial value of the dict on process start
// so that the call to "reset" and reset it accordingly
const main = async () => { initialDict = await maxApi.getDict(DICT_ID); };
main();


