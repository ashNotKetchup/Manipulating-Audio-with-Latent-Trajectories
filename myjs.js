const path = require('path');
const maxApi = require('max-api');
const DICT_ID = "myDict.dict";
const OLD_ID = "old.dict";

// Destination settings
const PROTOCOL = "http://";
const HOST = "127.0.0.1";
const PORT = 5000;

const URL = `${PROTOCOL}${HOST}:${PORT}`;

// Used for storing the initial value
let initialDict = {};

dummy_json = {
        "id": 2,
        "name": "Test Object",
        "tags": ["alpha", "beta", "gamma"],
        "scores": [10, 20, 30, 40],
        "position": { "x": 12.5, "y": -3.7 },
        "items": [
            {"id": 1, "label": "One"},
            {"id": 2, "label": "Two"}
        ]
    };

// Getting and setting dicts is an asynchronous process and the API function
// calls all return a Promise. We use the async/await syntax here in order
// to handle the async behaviour gracefully.
//
// Want to learn more about Promised and async/await:
//		* Web Fundamentals intro to Promises: https://developers.google.com/web/fundamentals/primers/promises
//		* Promises Deep Dive on MDN: https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Asynchronous/Promises
//		* Web Fundamentals on using async/await and their benefits: https://developers.google.com/web/fundamentals/primers/async-functions
//		* Async Functions on MDN: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function

// form for sending messages to python server and receiving replies
const send = async (message) => {	
	const res = await fetch(URL, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(message)
	});
	
	// python -> node response
	const reply = await res.json();
	console.log("Node received reply of type:", reply["type"]);
	return reply;
};

maxApi.addHandlers({
// Handle commands/messages from Max inlet
	test_set: async () => {
		// const oldDict = await maxApi.getDict(OLD_ID);
		const dict = await maxApi.setDict(DICT_ID, dummy_json);
		await maxApi.outlet(dict);
	},
	// reset: async () => {
	// 	const dict = await maxApi.setDict(DICT_ID, initialDict);
	// 	await maxApi.outlet(dict);
	// },

	// types of messages are 'request_latent' or 'encode' and 'sending_latent' or 'decode'
	request: async (filepath) => {
		// Node -> Python POST request

		message = {
		    "type": "request_latent", 
		    "content": filepath
		}
		
		reply = await send(message);
		data = reply["content"];

		if (reply["type"] == 'latent' ){
			dict = await maxApi.setDict(DICT_ID, data);
			await maxApi.outlet(dict);
		}
	},
	
	send: async () => {
		const dict = await maxApi.getDict(DICT_ID);
		let json;
		try {
			json = JSON.stringify(dict);
		} 
		catch (err) {
			console.error('Failed to stringify dict for sending:', err);
			json = '{}';
		}

		const prefix = 'sending_latent';
		const message = {
			"type": prefix,
			"content": json
		}

		reply = await send(message);


		// Feedback via Max outlet
		const new_dict = JSON.parse(json)
		await maxApi.outlet(new_dict);
	},
}
);











// We use this to store the initial value of the dict on process start
// so that the call to "reset" and reset it accordingly
const main = async () => { initialDict = await maxApi.getDict(DICT_ID); };
main();



