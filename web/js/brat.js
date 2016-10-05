var bratLocation = 'http://weaver.nlplab.org/~brat/demo/latest';
head.js(
    // External libraries
    bratLocation + '/client/lib/jquery.min.js',
    bratLocation + '/client/lib/jquery.svg.min.js',
    bratLocation + '/client/lib/jquery.svgdom.min.js',
    // brat helper modules
    bratLocation + '/client/src/configuration.js',
    bratLocation + '/client/src/util.js',
    bratLocation + '/client/src/annotation_log.js',
    bratLocation + '/client/lib/webfont.js',
    // brat modules
    bratLocation + '/client/src/dispatcher.js',
    bratLocation + '/client/src/url_monitor.js',
    bratLocation + '/client/src/visualizer.js'
);
var webFontURLs = [
    bratLocation + '/static/fonts/Astloch-Bold.ttf',
    bratLocation + '/static/fonts/PT_Sans-Caption-Web-Regular.ttf',
    bratLocation + '/static/fonts/Liberation_Sans-Regular.ttf'
];

var dispatcher;
var conf;

function initBrat(msg) {
    alert("initBrat");
    if (conf === undefined) {
        conf = JSON.parse(msg.data);
        head.ready(function() {
            dispatcher = Util.embed('view', conf, {}, webFontURLs);
        });
    }
    //if (dispatcher === undefined) return false;
    //return true;
}

var testdata = {
    // Our text of choice
    text     : "Ed O'Kelley.\nAAA BBB",
    entities : [
        ['T1', 'Person', [[0, 2]]],
        ['T2', 'Person', [[3, 11]]],
        ['T3', 'Person', [[13,16]]]
    ]
};

var testcoll = {
    "entity_types": [
        {
            "type": "Person",
            "labels": [ "Person", "Per" ],
            "bgColor": "#7fa2ff",
            "borderColor": "darken"
        }
    ]
}
