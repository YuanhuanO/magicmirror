/*Module.register("helloworld", {
	// Default module config.
	defaults: {
		text: "Hello World!"
	},

	getTemplate () {
		return "helloworld.njk";
	},

	getTemplateData () {
		return this.config;
	}
});*/

Module.register("helloworld", {
    // Default module config.
    defaults: {
        text: "Please get closer to activate the Magic Mirror.",
    },

    getDom: function() {
        var wrapper = document.createElement("div");
        wrapper.innerHTML = this.config.text;
        return wrapper;
    },

    // Add this function
    notificationReceived: function(notification, payload, sender) {
        if (notification === 'HELLOWORLD_UPDATE') {
            this.config.text = payload.message;
            this.updateDom();
        }
    },
});

