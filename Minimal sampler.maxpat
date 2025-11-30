{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 59.0, 119.0, 640.0, 480.0 ],
		"bglocked" : 0,
		"openinpresentation" : 1,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"comment" : "Sample FilePath",
					"id" : "obj-128",
					"index" : 2,
					"maxclass" : "inlet",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1175.806460022926331, 41.935484170913696, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"comment" : "Formatted Midi Events",
					"id" : "obj-127",
					"index" : 1,
					"maxclass" : "inlet",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 356.935486316680908, 41.935484170913696, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-125",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 189.90322732925415, 72.580645680427551, 70.0, 22.0 ],
					"text" : "loadmess 0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-120",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 189.90322732925415, 38.709677696228027, 150.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 10.317460477352142, 219.047622442245483, 62.698413670063019, 20.0 ],
					"text" : "test data:"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "",
					"comment" : "Audio Out (R)",
					"hint" : "",
					"id" : "obj-118",
					"index" : 2,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 333.870970129966736, 1595.887108206748962, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"annotation" : "",
					"comment" : "Audio Out (L)",
					"hint" : "",
					"id" : "obj-115",
					"index" : 1,
					"maxclass" : "outlet",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 296.774195671081543, 1595.887108206748962, 30.0, 30.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-107",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 64.516129493713379, 493.548390626907349, 58.0, 22.0 ],
					"text" : "loadbang"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-103",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1251.612912178039551, 280.645163297653198, 70.0, 22.0 ],
					"text" : "vibes-a1.aif"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-97",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1216.129040956497192, 250.000001788139343, 74.0, 22.0 ],
					"text" : "sample path"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-92",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1219.354847431182861, 316.129034519195557, 95.0, 22.0 ],
					"text" : "prepend replace"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-66",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 779.03226363658905, 622.580649614334106, 58.0, 22.0 ],
					"text" : "loadbang"
				}

			}
, 			{
				"box" : 				{
					"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-61",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1469.354849219322205, 1072.580652832984924, 107.0, 20.0 ],
					"text" : "s ---SampleChannels"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-123",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 285.064518332481384, 143.548388123512268, 29.5, 22.0 ],
					"text" : "- 20"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-121",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 252.806453585624695, 109.677420139312744, 50.0, 22.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-117",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "float", "float" ],
					"patching_rect" : [ 189.90322732925415, 277.419356822967529, 115.0, 22.0 ],
					"text" : "makenote 124 1000"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-114",
					"maxclass" : "newobj",
					"numinlets" : 7,
					"numoutlets" : 2,
					"outlettype" : [ "int", "" ],
					"patching_rect" : [ 189.90322732925415, 340.322583079338074, 82.0, 22.0 ],
					"text" : "midiformat 0"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-108",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 189.90322732925415, 201.612904667854309, 54.0, 23.0 ],
					"text" : "+ 48"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-109",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 189.90322732925415, 225.806453227996826, 62.0, 23.0 ],
					"triscale" : 0.9
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-110",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 189.90322732925415, 112.903226613998413, 24.0, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 22.0, 243.650797426700592, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-111",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 189.90322732925415, 174.193549633026123, 94.0, 23.0 ],
					"text" : "drunk 36 12"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-112",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 189.90322732925415, 143.548388123512268, 82.258065104484558, 23.0 ],
					"text" : "metro 150"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"format" : 4,
					"id" : "obj-165",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 189.90322732925415, 254.838711500167847, 48.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 14.0, 269.650797426700592, 48.0, 20.0 ],
					"prototypename" : "Live-MIDI",
					"triscale" : 0.75
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-62",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 189.90322732925415, 309.677421569824219, 115.0, 20.0 ],
					"text" : "pack"
				}

			}
, 			{
				"box" : 				{
					"bubble" : 1,
					"bubbleside" : 3,
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-180",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 183.87096905708313, 1598.387108206748962, 110.0, 25.0 ],
					"text" : "Turn on audio"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-116",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 245.16129207611084, 914.516135573387146, 45.0, 22.0 ],
					"text" : "open 1"
				}

			}
, 			{
				"box" : 				{
					"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-106",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1469.354849219322205, 1045.161297798156738, 96.0, 20.0 ],
					"text" : "s ---SampleLength"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-105",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "bang", "" ],
					"patching_rect" : [ 1340.322590231895447, 954.838716506958008, 31.0, 22.0 ],
					"text" : "t b s"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-104",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1340.322590231895447, 1064.516136646270752, 107.920792400836945, 22.0 ],
					"text" : "21362.358277"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-101",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 10,
					"outlettype" : [ "float", "list", "float", "float", "float", "float", "float", "", "int", "" ],
					"patching_rect" : [ 1340.322590231895447, 988.709684491157532, 113.5, 22.0 ],
					"text" : "info~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-100",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "bang", "" ],
					"patching_rect" : [ 79.032258629798889, 564.516133069992065, 62.376237809658051, 22.0 ],
					"text" : "t b s"
				}

			}
, 			{
				"box" : 				{
					"bubble" : 1,
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-18",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1353.225816130638123, 595.16129457950592, 75.0, 25.0 ],
					"text" : "read file"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-26",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1114.516137003898621, 419.354841709136963, 131.0, 23.0 ],
					"text" : "replace drumLoop.aif"
				}

			}
, 			{
				"box" : 				{
					"bubble" : 1,
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-29",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1314.516138434410095, 717.74194061756134, 154.0, 40.0 ],
					"text" : "save buffer~ contents to audio file"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-33",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1267.741944551467896, 729.032263278961182, 39.0, 23.0 ],
					"text" : "write"
				}

			}
, 			{
				"box" : 				{
					"bubble" : 1,
					"bubblepoint" : 1.0,
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-41",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1662.903237700462341, 608.064520478248596, 175.0, 40.0 ],
					"text" : "set waveform~ to display contents of helpbuffer-1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-51",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1225.806460380554199, 885.483877301216125, 185.0, 36.0 ],
					"style" : "helpfile_label",
					"text" : "Position (ms) of mouse when dragging in buffer~ window."
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"format" : 6,
					"id" : "obj-56",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1227.419363617897034, 864.516135215759277, 99.0, 23.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-59",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 1546.774204611778259, 554.838713645935059, 67.0, 23.0 ],
					"text" : "loadbang"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-72",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1546.774204611778259, 616.129036664962769, 122.0, 23.0 ],
					"text" : "set ---samplerbuffer"
				}

			}
, 			{
				"box" : 				{
					"bgcolor" : [ 0.501961, 0.717647, 0.764706, 1.0 ],
					"buffername" : "---samplerbuffer",
					"gridcolor" : [ 0.352941, 0.337255, 0.521569, 1.0 ],
					"id" : "obj-68",
					"maxclass" : "waveform~",
					"numinlets" : 5,
					"numoutlets" : 6,
					"outlettype" : [ "float", "float", "float", "float", "list", "" ],
					"patching_rect" : [ 1545.161301374435425, 659.677424073219299, 330.0, 135.594054999999997 ],
					"selectioncolor" : [ 0.313726, 0.498039, 0.807843, 0.0 ],
					"setunit" : 1,
					"waveformcolor" : [ 0.082353, 0.25098, 0.431373, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-75",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1279.032267212867737, 787.096779823303223, 44.0, 23.0 ],
					"text" : "open"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-77",
					"linecount" : 3,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1438.70968770980835, 885.483877301216125, 88.0, 50.0 ],
					"style" : "helpfile_label",
					"text" : "bangs when done reading or writing"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-81",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1500.00001072883606, 862.903231978416443, 20.0, 20.0 ]
				}

			}
, 			{
				"box" : 				{
					"bubble" : 1,
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-83",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1319.354848146438599, 656.45161759853363, 187.0, 40.0 ],
					"text" : "read audio file and resize buffer~ to contain entire file"
				}

			}
, 			{
				"box" : 				{
					"bubble" : 1,
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-86",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1330.64517080783844, 775.806457161903381, 147.0, 40.0 ],
					"text" : "see contents (or double-click buffer~)"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-87",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1227.419363617897034, 598.387101054191589, 119.0, 23.0 ],
					"text" : "read drumLoop.aif"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-88",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1243.548395991325378, 632.258069038391113, 109.0, 23.0 ],
					"text" : "read vibes-a1.aif"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 13.0,
					"id" : "obj-90",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "float", "bang" ],
					"patching_rect" : [ 1227.419363617897034, 832.258070468902588, 170.0, 23.0 ],
					"text" : "buffer~ ---samplerbuffer 100"
				}

			}
, 			{
				"box" : 				{
					"border" : 0,
					"filename" : "helpargs.js",
					"id" : "obj-91",
					"ignoreclick" : 1,
					"jsarguments" : [ "buffer~" ],
					"maxclass" : "jsui",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1525.0, 830.0, 190.398513793945312, 84.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "float", "bang" ],
					"patching_rect" : [ 1225.806460380554199, 543.548390984535217, 134.0, 22.0 ],
					"text" : "buffer~ ---samplerbuffer"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 64.516129493713379, 593.548391342163086, 72.0, 22.0 ],
					"text" : "prepend set"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 64.516129493713379, 533.87097156047821, 94.0, 22.0 ],
					"text" : "---samplerbuffer"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-9",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 285.483873009681702, 517.741939187049866, 32.0, 22.0 ],
					"text" : "0. 1."
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-95",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 377.419357538223267, 1166.129040598869324, 44.0, 18.0 ],
					"text" : "Version"
				}

			}
, 			{
				"box" : 				{
					"bgmode" : 0,
					"border" : 0,
					"clickthrough" : 0,
					"embed" : 1,
					"enablehscroll" : 0,
					"enablevscroll" : 0,
					"hidden" : 1,
					"id" : "obj-96",
					"lockeddragscroll" : 0,
					"lockedsize" : 0,
					"maxclass" : "bpatcher",
					"numinlets" : 0,
					"numoutlets" : 1,
					"offset" : [ 0.0, 0.0 ],
					"outlettype" : [ "" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 6,
							"revision" : 5,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 1868.0, 202.0, 10.0, 10.0 ],
						"bglocked" : 0,
						"openinpresentation" : 1,
						"default_fontsize" : 10.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial Bold",
						"gridonopen" : 1,
						"gridsize" : [ 8.0, 8.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"boxes" : [ 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-10",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 40.0, 368.0, 191.0, 20.0 ],
									"text" : "prepend script sendbox Max6Warning"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-9",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 56.0, 304.0, 41.0, 20.0 ],
									"text" : "10. 10."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-7",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 40.0, 280.0, 52.0, 20.0 ],
									"text" : "571. 169."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-5",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 3,
									"outlettype" : [ "bang", "bang", "" ],
									"patching_rect" : [ 40.0, 256.0, 50.0, 20.0 ],
									"text" : "sel 0 1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-4",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 40.0, 336.0, 123.0, 20.0 ],
									"text" : "presentation_size $1 $2"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-11",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 72.0, 408.0, 67.0, 18.0 ],
									"text" : "Thispatcher"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-8",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 112.0, 304.0, 57.0, 20.0 ],
									"text" : "hidden $1"
								}

							}
, 							{
								"box" : 								{
									"bgmode" : 0,
									"border" : 0,
									"clickthrough" : 0,
									"embed" : 1,
									"enablehscroll" : 0,
									"enablevscroll" : 0,
									"id" : "obj-3",
									"lockeddragscroll" : 0,
									"lockedsize" : 0,
									"maxclass" : "bpatcher",
									"numinlets" : 0,
									"numoutlets" : 0,
									"offset" : [ 0.0, 0.0 ],
									"patcher" : 									{
										"fileversion" : 1,
										"appversion" : 										{
											"major" : 8,
											"minor" : 6,
											"revision" : 5,
											"architecture" : "x64",
											"modernui" : 1
										}
,
										"classnamespace" : "box",
										"rect" : [ 2004.0, 234.0, 297.0, 97.0 ],
										"bglocked" : 0,
										"openinpresentation" : 1,
										"default_fontsize" : 10.0,
										"default_fontface" : 0,
										"default_fontname" : "Arial Bold",
										"gridonopen" : 1,
										"gridsize" : [ 8.0, 8.0 ],
										"gridsnaponopen" : 1,
										"objectsnaponopen" : 1,
										"statusbarvisible" : 2,
										"toolbarvisible" : 1,
										"lefttoolbarpinned" : 0,
										"toptoolbarpinned" : 0,
										"righttoolbarpinned" : 0,
										"bottomtoolbarpinned" : 0,
										"toolbars_unpinned_last_save" : 0,
										"tallnewobj" : 0,
										"boxanimatetime" : 200,
										"enablehscroll" : 1,
										"enablevscroll" : 1,
										"devicewidth" : 0.0,
										"description" : "",
										"digest" : "",
										"tags" : "",
										"style" : "",
										"subpatcher_template" : "",
										"assistshowspatchername" : 0,
										"boxes" : [ 											{
												"box" : 												{
													"alpha" : 0.8,
													"autofit" : 1,
													"data" : [ 1678, "png", "IBkSG0fBZn....PCIgDQRA...XI....YHX.....YakL7....DLmPIQEBHf.B7g.YHB..FTTRDEDU3wY6ctccaiCDFdPN66qJAlNPcfo6.mJXUpf3sBrRE33JvoCr2JfxUfbGHkJvpC92G.jkrLu.L.jDT5+6bzw2nvLBbvfAy.PKBgPHDBgPHDBgPHDBgPHDBgPHDBgPHDBgPHDBgPHDBgPhCyw+..lIhL28iuZLlcCuJQNq..E.3M7QdB.kistkq.fY.3F.rz8pXr0orC.bOZlJ1oc.2fvGqYf3aNu9j8.f0sXXsuS6lwVOGSbdnZa.H.vsisdlC7dLV..d9d9twX9c+nNeDbHluB2q+VNDC3o7h6qqDQ1ZLlsIVWlKh7XKxeOqLFy0oT1SZ5XT3ornmzgY.XArSyrIPc5T1.q2khDnWywmm1qIpRPWw4CJtwUlPYu.1EJzWToUeQXFU.vaO+0IqY.nDVuimGn3lUTAp55DWF5MsH4wPzYmNtITgDQ+www4tPS6jcn7FkJ29vNpL3aXIhGCPOqTz9aT1mTmG6o+hkf9azKBTNcsppdGO0yaU17AOXC1AZ0wzO8EvNMgF75COrt5qTJiTxadnq0krXe4dE88sEeYvsWNvWN56eowqpclIh3StadTDoToLRI+qGWych8ykFBpeD1Us11Td2hobxog0ihVZ0qErq5KGny3qf0akV5zaXMxymob8Ntvbg28X4J37yJamYR6i5JT1tohchMwte2iq8tHjil9u+wiqYAl3dsZJHReXcO0twRE7LuPHNu1.AlqLDl2wocohPboAnnk1cAF1bVsAguh0XlxVypACQdpRiQ1.hy6RqipvgcEPexSPYxEQbY+uTg7BM0KEZ9bkM.8oEvqfLg0.aIRSRReCNiIDYNefdOppBtFg2OOYlNzT2uD1QFqkvWxcvU12IqRwFf+Utecg7w.925ds+6+iHxqhcWttUR.N8PyzM6DQ9plcaKrqhLj93mMFy2BUNYEPYlmGa8dO3PAcK7750FBPYD5XnDb5LxRfh3gx.cdA97TLcVyMnyvZYD54bExC3bXmOfOWw8NYD00RzbLKcNEGB2vJpjVpPd6YQLxcn3Ks8GcwMbsXimIaA15oUIMWxnBj1h49aOS1ZePwHI2zC7u.xC5tmDg4Qszi1xGVlHcWqGqIwNTsUOV6wXL6bq16mcboClmMXCJuR5dOn6ENuyso+uJhbswXVlB4I5KH+zdazzDvlCppFFMULP5flc1YoGs6M079VidHtFXyimJRstzG7Wg9Fb4M5ZmQzBwl6ochHOj5SFSc.arRURODqgwXdF.eUrETemXySV1EeI.lk6mR8fMr1iyHZYxzD+4IIQS+UGtOW+psqwMnp3neUxOtYcvbwdL2xVTaXMF.6p+J07dMFyJkxrzIyqD6MzZiwwMC0VwFK1KhMK4aaooyNOgWjf3JNdPkpAMeD5CkJzP7YQ94oLE8oW7.kGCqiv2hiOC8ytunBmjwbDmg0hdoiNg3U5Fx.tShKX8N2G5v5EXiXWPRpoTDoJgFDEIpctbIxQ16o0b+fgaO4+FbojAwsaUWND88wvTviUL6AcQrAQ23Rygs.0C0gUXl37Hl6oKHVxZCKXm5nLxl4+53uOzmausM78mUj0FVR7dq14wibohHkQHrU93I4Yqx1I681ksFVtonJhrYdviqYUjxvW1Ih7sSlBTatrx9bfksFVhH+Hx2+NoiLn63aR+eiZeArOUN+omk6nQVZXgC6C9X3AeBP9n8bl1CqaW7KodiJQl.ddNq.w+DoQ0SoEXSsQP6X1Vj+ivic6gx1+7bqyz2f3OVXKhT9yg03NDirMvZLcSH23ghiZWLe1FJxthPCaoOJhnIVE6CeW2zVuOMENTatB4fts08RcAtc7hD1z9b5SM.8Ovy.NJy1SEP3UV3owVm8gbL38q59RZjeNv6Kpnw4sKj7Ro84X1fRNZXUn788rwX7I8B4HgrhzU8kRbVixo.WiI7Jkf+Gd0yiSB8XfBipIWbU0A7a0gStmreYCJLpl9G4bw6f3m9OdtGKtDMp1CZ2q0z9Au1XimFUqwYvzemBZ2q0xwV+lzfty598XBGndWf5Km0z+ej.iMn4CyvFbAb5TP8OOJXrUwRMcr8xQbOmAG9GXUEMpHDBgPHDBgPHDBgPHDBgPHDBgPHDBgPHDBgPHDBgPHDBgPHDxEI+OU18UDTuSZQA....PRE4DQtJDXBB" ],
													"embed" : 1,
													"id" : "obj-21",
													"maxclass" : "fpic",
													"numinlets" : 1,
													"numoutlets" : 1,
													"outlettype" : [ "jit_matrix" ],
													"patching_rect" : [ 64.0, 16.0, 40.0, 32.0 ],
													"presentation" : 1,
													"presentation_rect" : [ 232.0, 41.0, 52.0, 40.0 ]
												}

											}
, 											{
												"box" : 												{
													"bgcolor" : [ 0.75, 0.75, 0.75, 1.0 ],
													"bgoncolor" : [ 0.75, 0.75, 0.75, 1.0 ],
													"fontsize" : 14.0,
													"id" : "obj-20",
													"legacytextcolor" : 1,
													"maxclass" : "textbutton",
													"numinlets" : 1,
													"numoutlets" : 3,
													"outlettype" : [ "", "", "int" ],
													"parameter_enable" : 0,
													"patching_rect" : [ 24.0, 88.0, 136.0, 24.0 ],
													"presentation" : 1,
													"presentation_rect" : [ 71.0, 48.0, 144.0, 24.0 ],
													"rounded" : 20.0,
													"text" : "Try Max 7 for free",
													"textcolor" : [ 0.258824, 0.258824, 0.258824, 1.0 ],
													"texton" : "Try Max 7 for free",
													"textoncolor" : [ 1.0, 1.0, 1.0, 1.0 ],
													"textovercolor" : [ 0.258824, 0.258824, 0.258824, 1.0 ],
													"usebgoncolor" : 1,
													"usetextovercolor" : 1,
													"varname" : "live.text"
												}

											}
, 											{
												"box" : 												{
													"autofit" : 1,
													"data" : [ 169659, "png", "IBkSG0fBZn....PCIgDQRA..B....H..HX.....83Qs9....DLmPIQEBHf.B7g.YHB..f.PRDEDU3wI68dGkeTcmmnetU2sjZI0pUDkinDBkPHPBSP.FDfDf.LhLFYOoyaede6N6Nu2ry91cdmyt647l2YFONOd7LdlwiSLfbfLXL1FvDDAoVnbqDHoV4brkTq9W89ip9lt0s90A0JWe0A5tqp9d+Ft25a5FJGJfBn.NeF5zblyW3pPmq3pqLpxw6hhGYjKZ.tXziN04N2it08Z5V25Z0cpSctSU14tzknNWUknhppxUYEUhJpnBzopp.UVQkv4hfKxgH5mQQHxADCfRkhQoRwHNtDhKEi33XzbymBmp4lQSmpYT5TMilN0IQSM0T7oZ5TwMdxi27wN5wa5HMdriezibzC0byMeDGvdckPCmJFq00zoVwINw9+nW+0e8sbtV4U.EPAjO3NWy.EPAboJLqYMq918Kaf2aknpaHNNdzQU3FRe5cu6WOqsGUWaM0D0st2cWW6Z0nacsqnacqan6cuanqcqqn6cqan5tzETYkUdtVDPbbIbhSbRbzFaDG4HGAG6nMhibzifibjihFa7X3XM1HN5AORoCc3C2zt10dN3IN0w2QoRweVympzxOQoC8Ju9y+Ju24ZYn.JfKUgh..JfB3LHL6YO6gVQW66STUDt4JpzM190uAz+90ud24ZqsGQ01idfd2qdgd0qdhZqsFziZ5ApnhJNWyxmQgi0Xi3fG7P3fG7PXe6ae3.G3.3PG5v3.6+.mZG6dmG7XG8XaNtj6iOwwNwy+JuxBe4y07aAT.WLCEA.T.EPG.LiYMuQbY8t5+vJqphaqaculwLvALfZ6ae6cE8tW8BWVe6C5W+5K5Se5yE8N3OcgCejif8rm8hcs68f8tm8f8su8Gu68t+Srsss0sVpjaImxc7m8kW3B+Ymq4yBn.tX.JB.n.Jf1HbqOvCbK0VQm+SbUT0LF1PF7f6+k0up5a+5GF3.5OF3.F.ps1dbtlEunCZpolvN20tw11wNvN2wtvt2ydh21129gN3ANvZNUboWXkK9C9Nabia7fmq4yBn.tPBJB.n.Jfx.21scaSp60zu+CQcoS27vGxPF1.Gv.pZfCr+XHCYPXfCX.mWLO7WJCG3fGDacqaCMr0shsuycEu0FZ3P6a+GboMcpld1W4W9reO.z74Zdr.JfyWgh..JfBPA2y8L+GMpKc5Op28tWSeTiX30L3AMHLzgNDL3AMPzoN0oy0rWAzJf8ru8gF1xVwVZXqXKMzPye1l2xmcpS1zqTeSG4up9W3E114Z9q.JfyWfh..JfKog4N+G4KWIp7OYvCdPScDCencZHCYvXjCeXne8sumqYsBnCBN9INA17VZ.adyaAMzvVi23F1vtN5wa5UZ7f67+mhspXAboLTD.PAbIEbmOzCcOUGW0eV+5+.tlKejCu5QL7ghK+xGE5Ys0dtl0JfyRvoN0ovV1xVwF13mhsr4sTZca7S25wO9wetUT2h9uWrNBJfKkfh..JfKpggL6Y26oVae++sm0Ty8O1wb48aTiXjXLi9xQe5SuOWyZEv4IPSM0D13msIrwMrQrgM9oM8oaZK0U53M8+2K9hOyu3bMuU.EvYRnH.fB3hN3tu66a1UTc2+eNxQN7oc4ibDUN1QOZLhQLrhsfWAzpfCbfCh5W65vF9zOEqcsqeuG5PG9m9Baci+4XQKpwy07VAT.cjPQ..EvEEv8ceO5Woxt14+iicridTieLiwcEWwXQu5YuNWyVmQglKUBGuwigFa7XnwFONN1wNJNwINAZpoSglN4IQSM0T5e2DZt4lQoRMmdL+VBwwkPoRw..nhJp.NmCQQN3hhPEUTIppxJQm5TUnxp5D5TUUhN0oNip5TmP0UWMpt5pQW6ZWQ0UWM5Tm574XsvYVn4laFqeieJVS8qE0W+ZabG6Z2+pctqi7m9Au0y+Ymq4sBn.Ncgh..JfKXg68Adr+a0Ta2+2OgwOtKariYLX7iaLn5pq9bMacZAM0TS3.6e+X+6e+3.GXe3.G3f3vG4v3HG9v3vG9v3HG9H3vG9f3nG8XnwSbbDAGhiigyAPuNq+6jeO45kPLhfCIeA.b7ee5fekQUfpqtZTSOpA0z8dfZ5QMn6cu6nlZ5ApoG8H4TNrm8F8tW8Bcu6cCNWzYOk4Y.ngstMr50TOVS8q+Te5l1z6rObh+2emEtvUctluJfBn8.EA.T.WPA2088X+E8p1t8e3Jmv36+UbEiGWw3FKppppNWyVsZnolZB6cO6A6YO6A6YO6B6d26A6cu6A6dO6AGX+6CG6nGCjCVAj+14RbPC3.6KVC5qE39NWhS8DzC+6kG+1.8U7ebLPkUVI5YO6E5cu6M5W+5G5Se6K5ae6C5a+tLz291Ozidbg0Anzt10tvxWwpvppesm5S2vVdqCt2i9Udq25Wrly07UAT.sVnH.fB37dXdy6gWPmqo6+OlxDmvPuhqX7X7iaLm2um7O7QNB1912F14N1A14N1A1wN1N1wN1N12d2GJUpDbtjLncw.N5yxWN.kANkYdbL3eR2WbjmdckiXFeDm3nW8SDmhe50H5DD+1B8MBP3qyxeptnqU2Uz+92eLfALPLfAN.Lf9OPLfAL.zu92eTQz42UNXG6bmIACrp5aZSMrkW9f6pg+n25sdq8btluJfBnbPQ..Ev4kvce2Oz0TY0c56N9wOtoMoINA2TlzUhtzktbtlsx.MWpD14N1NZngFv11ZCXKaYKngs1.NzAz6lr7yXNw4ZbpS0TGoP6TleRaakyu6WNeKwTnvNuUNlMQOngxSy7nOw+RACjoTPB9nkoekUVAFzfFBF7PFLF7fGBFxPGBFxPFJ5V25dVY77.XKMz.V5mrBrrUrhCu+8t2u2y8yd5+OOWySEPADBJB.n.NuAlvDlP2GyDt5+4gOrAeeSbhWYUW0Tm34UKju33RXm6ZWXyaZS3y9zOEaZSeF17l2LZ5jmjyRF.dNZy5PibzSnPN+MzhxRWMG711OOGw9zOqSVmxQsw4K+C8zADHffVK8ALN7c7MQX5mxqsH8Swuu8seX3iX3X3CejXDibDXXCa3mWEjXyM2LV65VOV1xWA9jksxsbfFa7+iW+W7zO24Z9p.J.BJB.n.NmCy8AehGu6cqy+0SaxSYfW8zlJFwvG14ZVB..m7Dm.e5msQrg0uArgMrd7oabinwFaDkR8tEEHqdcF8IW.bo0IHNEAWP7yVpbpMoGVW5c.kSVNXAphBoLfG36LlydWMM.ZN0g1G8yHSAWe.goeLSDXm1hLJGnKw.FP+GHF8XFMF0kOZL5QOZzu9cYgH3Yc3XM1HV5RWFpqtk07mtoO8kW1hWzWr3PGp.NWCEA.T.mSfgL6Y26o0899zicbi91tpoL4noL4IdNO6sFa7XXcqcsn95qGaX8qGaYKaFkJUJc0uCnSs1QSauJCa+x4aAalr14PWvO6T.jsb4511ed8Y5Sqq.1oa9zWtH7DpjqEG2Nnuq7Ns0xUP5CKMLUxnEvWW0jZpoFb4idzXricbXbie7XPCZv3bMrosrErj59D7I0srce3Cbv+ud9m+Y9Amq4oB3RSnH.fB3rJLm4L+4Vaep86cUSapCdFWyUiAOnAdNiWN4IOAV+5VOpu90f5qeMXyaZSHNNNwgO6gOELYxGab3W14aGZeo93qeXAx5T06wBtx78ulNXh7nucN446lIXj.U7uEoOPoX+JMXkRYMCnVu.xsJ6Biz9Psd7qolZvXG23v3F23w3F+3wkcY8ubD4LJbxSdRT2RWF9vOdwMuwMs4E9hK7m7T.3jmyXnB3RNnH.fB3rBLmG5Q+1CafC4O5Zu5o1ooe0SCc8bz90e6aeaXkqXEXUqZkXs0WexAjSHG9vKybkCFwolxqn1gi2bumG9l42uLN7zNiybMlV1pG3WV8fzWk0NWdeXK4edYwWN5mU9yqZC9k9WnucGGT9R+Gr5CsD9HoRA8qe8CS3JuRbkSbhXri6JPW574lC1nM9oeF9vO9iQc0s75OwQZ7IeoW5Y+nyILRAbIETD.PAbFCFxrmcumVOtrmaxS7JtgYNiq0Mtwd4m0OHXZpolvZV8pwxV1mfUsxUf8su8k8gT9.0YbGxAsuSsPGXNsF7YRqvurzO.3Om4NmilJbISXOYrkbF2R32hzOy8QNAUn3iPzONe4WiuLULsF7E5mQVhiQUUUEF8nGCtxINQLkoN0yIqefCcnCiO7iWL9fObwG9f6ZG+Y+hmag+Cm0YhB3RFnH.fBnCGtw6+9m1.6Zs+roOsqdjW2m6Zw.6+Y2xrdzidDrrksLrrOYoXkqXEnolZB1R0SyiueVyv64zNXnK.1QJ8bYW7bvfO6fWgullV7sk1NC8M7uuSTEO6EPSF56MO5gVSCAwuUP+vP1r6QKQeDP+CXvW+29qoBK9V5mbuvUF.H4LJXPCXPXpW0TwTl5UggO7geVM30lZpIrjktL7AevGep02vm88dkm8o+Jm0HdAbICTD.PAzgA2y8L+GsOC7x9ty3Zldsy7ZmNpolZNqQ6CcnCgkT2RvR9nOBqa8qMnCSf7c5lTldXbNqcNjal3dADjmyk71lelLyUz2DO.0dYx3VnoVtRtp3QVx.OrNQ+6sV7yk9dKhQqW8vNzyi9V8mMzHa.YsL9RCo3+.UuvOPI5vJpm8rmXJS8pvzm90fQM5QeV6fIJNtDVS8qGu26sn3Uut5eimaSaXdEeThJfNJnH.fB3zFtu66Q+JW1fGve80ecWaWtloe0m0Nk9N7gOLVxRVLpawKF0u1034gP6zzlQadYrqyJMuL5kmM6hjqbkqOaCn4SuoS.gV7bhGuPyYtekKNSBsc5GmI677x9N3TTj1F7u0NvOyBZLWYBF8ulFz2BgZpoFb0Se53pu5qAW9nO6MsVMr0sg29cdWTWcK8C9nCr64zvq+5AlOqBn.Z8PQ..EP6Ft6G3w+KG9PG7+sO20Oypt5oNYTYkUdFmlG+Dm.eRcKAevGrHrlUuZyBQSUa6LYx6b1RVGxYuszx11JaIqC8pSfrpyEe.+W+LNdZEAgbgBD57EHjrmYJL.BO0Fs.9dQ7EXpIrASIWRUIAuJbvMkWa0yd1KbMW60hYLyYhAO3gTV8PGErycsK7Nuy6gO3iV7JOx924c85u9qukyJDt.tnCt3vBSAbVEt2G3w+qFwHFxe1MciWeESZRS7Ld4PatTIT+ZVM9vEsHrjkr3z4zGvetn8mGexwieVm4dX4DLvAXZqxsv7Bev33g+EQN1aeP3JEDZcPDZZFhiKk9y1N9jGbpuQuUL0aqyP6bA+9+jJCHxRIDigMjggYLyYhq4ZtVTaO6YGftp7vd229wu+ceO7wK5i2vmts8caEehhKf1JborknBnMB28C73+kibDC6+9MeSWekSdRS3LdoO28t2Ed224cvhd+2EG7fGxdSpb9gbHqqHsWF+.4O+6TFdYW43Btgxn2+CryYaG7YJGdn8eWYtV6F+yJf+hh.ABrxikLY8ql5.3s1.BTc.+sFYvsOn25LPqRJgXTgKBSXBWIt9a3Fwjm7jQEmgqL1ANvAwa9VuCVzG+QqY4Gdee95egWXamQIXAbQCTD.PAzhvb+BO7e9HG1v+eNqa7yU0jm7jOilweSM0DVZc0g28ceaT+Zp2r81zN0sKJNak.XP8r9P18JNgP3rICt2xMYEJ3el.BsnB8cZybb.G7z0MG9OZm+sS7KGO1w.gpXfsBOgW3ggvWGUnt+zN9IyZ.PEroskCPeuJCz8t2cLyO2mCW+MbiX.8e.cLpjbf8tu8i25sdGrnO5CWwV+z0daevG7A67LJAKfK3gh..Jfbg4MuGdACZ3C6u+luoOWmm5TmxYTG+6ZW6Bu0a96vhd+2CG6XGEgl+1Lq57blqeiQ+.UAn0T0.+.BR9qPAHz9A6gmiMab6wsKx3HlbXGtf5YgPtAaKPd3a9ch+8Zeh+YdtUrM7ZePXNK65KHDGlMvFc.kAWXfdiOybnIw2OYZEFyXGKtwYcy3pupocFsp.6Yu6Eu4a91XQeXcK541xZu0hcMPAjGTD.PAjAty6b9yZ.C6xd9YcCetZutYNiyXKtulKUBqbEKCu0a9lXEqbEIebcHvy4NP9Nd82hbgtuF+Xj5L0eglQwTnmOXfba2VM343i4YeGgpmS34zWRib.khsWi.08L+t59Q.nj954gSaDeaN0HC+EL.A+JHDZpF7g10TOX6+LkqOGm2AWC.oiKRNZiy6fLxmpvL9R+70TSM3FtoaB2zMcynmmAWq.aa66.uwa7awmrhU8yd9E9im+YLBU.WvBEA.T.LL84N2QM59Nf27Futqan2zMcCn5yPebdN1wNFdme+ai29sdSr28tW4FAb5afLk+mtHgitbtANpaC3oRpXPRaEdK305fPN08yHtkx9lteTjKwgqxgb3qEAfXTBIecBKUpTpCaG+QLJ4dIxDsU1z3b5gex8QoXg+xgWKGDZ5Hxd7+1dBBHCk.0OyikTY7m86WPrDvffNmYO8GxhLsD76g4EfHgRJDE4vTupoga4V+7XzidLmlxU9v52vFva7FuYo0rwM927RK7m7meFiPEvEbPQ..E..Pmt+G5we6q65l4LtsaYVnG8nGmQHxd1ytvu4M9M38eu2EG+3G2tp8SyH2jstor3IOo+7QavGfMdab96ksu+hGrCag64MO5Zm+skxtGEEgRkJk4mHJoEHG0QoSIi3TVCITjtm8KZnEmSG7KgXw4OTAJDh+Y4yYqlPYfPUKHXvAsQv+v+gGh30gEWJlClThiz5nW5qyt1AnnGrSCgF+jfGFwHGI9729rwzl1UeFYp1hiKgks7Uh2327lmrgOayK3EewE9zc3Do.tfCJB.3RbXNOzi9smzXuh+cy8Nuc2fNC8k4aiabC3M90+Zrz5VRlxfCXMfF4uEub4WB9P3m4YY7syIb6IKe4zBTxxWSe.DLiWxgG4jzl8txYpxoNR4T.Ghc.NSSlNU.tjRyGGC9YbtjLx0OudgNpuWrCcP3qxD14T3p4e4YJEP9QpdwWGERm5e7MSUM.pftZaUJPbRqOfiDG11ly+rffb16e5Blfrs7+93qCLnDhQe6cevs94uMb82vMdF4yi8oN0ovu+cee7lu0auqc1vmdCuwa7FqqCmHEvELPQ..WhB207d3Gd3Can+vYe62Rml7DuxyHzXkqX43Ue0WAqa8qK04hjQdvETWNSAPFG1pfBJO9JGSH+.IBBg11a5UEef4.uBclso22j8KeMG6nzlgsH.z8K2ThCXYQmywGcsZ4mbFocxaKc8oG95qGm54LO9mjSBG+fejJJHSsQvoPvW+667OT+WqBjx46K+Ae5fqq.a8dZq3GGGip6ZWwMey2Lt0a61QMcui+H09vG9v3W+FuId2E8Auyyuve7mGEeFhujDJB.3RLXb268NnqtOC7iu4a7FF3McCWWG9B7KNtDpaI0gW60dEr4MsIii7PeY1Rtt4uZgL4U3ySaPnRr2N1S9JmEg1dbY.kCcJCe.j96T4ucY3LVB7bbpyzmOrZbTY0gpchs+sJvG49YKauG063vGP8rRvD97udMC3BF3PV5CjU9ofDZo0WPl0SPab5Cx6jYztJ+05gXk7KACkU+EqjWc0GrwODWJFU0oNga5ltIba29cfd0qd0p361BzvV2Fd4W80iWe80+W8K+4O8+0NbBT.mWCEA.bIDLuG7Ie1oO8oL+65NtczyZqsCssatTI7Qevhvq8puJ14N2g4dl8PORcjSFOMy8uMScclogpBfN86S2Usen4YFHr4a.XbvWxKiU+4SOSIjcIOi+4Jf3vKTfNlTLMqKhPzH4hdnWFd5zEehWyJ+RCnc7mU9y1d9Sc.GTfW.Vk7V.hVNRpPPqZ2FjC36fN64xPBMnwegks3bwmohsalgnJhv0MyOGty4bWnu8si+yT7hWRc3W8q+sGYWacm25K8RO6G0gSfB37RnH.fKAf4N+4+fibvi7mdW24rqZbiczcnscykJgk7weDdwW3Evt28tPl47zKycc15RoVA3qpd9Pk22fOzscdtpsfNydtb9gv1K6RxAujYryfEcMRbHmcz7l6+797rO9ljG8SVW4.gJGenCinLAc40fcz3mneKC+mw4lHX4Qe92SuLslExn+7p1BWkf.fNnf12ZFfZmvmM.TejNngPmZfV7KytGPwZQUTAttq6yg6ZtyE8o28oMyykCZrwFwq+a9c3ce2268+EOyO9FAPycnDn.NuCJB.3hXXTiZT0dsW2rV5sbK27Hl0Mc8cnk6OoT+KAu3K77Xa6X6lr9zY0lbAXqZcLxXrWuJr0y+pY0Yavu0Mz0rPsfM..SnC4r38zk2k2pbJm0RBk1RVyqHdk.3mcrIITkCRqOeIXHa.PpGzDXQb3FSo.8OJj6nvWNvjxm+0PdUQHSTO.p0l.0WF3jeTgel.BT6BgLgg0lCDv1B5yWh7pXhM6dK+G73L1GacvPwwnxJq.2vMNKbm20b5vOKAZngFvK9J+pR0ug0+e4kV3S+W2g13Ev4UPQ..WjB2y8+H+ul3UNg+q227taWe5Su6Pa6ku7kgm6W9Kv115Vy3PVezopAeG9x0yYK4o8GgVgC+Vnl14gusD9hiBY03mV5Z0hZKQNEYpDR18BHF4Ou4dA9j87r2SfU+soRGNiODUEAD9Jj5vR+yL3Ky4cY3e5pYjeabF51KzzpnwWedE3x3DMafa5pC3edEb5dJEFdE9qhbSw+rZpEtdlCPIUvTUVYk3ll0Mi6bNyE0z8t2l427flKUBKZQeD90uwuY6qa0exUUbrBewITD.vEYvLl07FwXt7At34bmyt2W6zmVGZa+oe5Fwu7m+ywZWa8dKtOuL94q6eX7.v4ExYhCnKur+BBrsBgNJZM.koe5B1CpEumIqNmcKxA.yBVipRgw4TFmyhb66VjrvKNG8xp0qsBUAjrUEP8.dATHnm0QRGE9L1ki+4VyJ+103gP+LkA2kbX83e.F4GP.f+VbTZ.+cbP4VLgRR64EYT4gPY2m8aHP9UxP6rO7WjvDnScoy3Nui6Be9a61Pm5Tma07WKA6+.6Gu3K8ZXkKe4eie9B+o+G6vZ3B37BnH.fKhfuv7e7+koNsorf6dN2EpolNtsNzN14NwK7b+RT2RVr45Y268ZK+Yy5OaY9oa.iChV5n8MC3s3tLtP7NI8.jxya2y5YA6hUykwQTvrmMWOPUMxA+3.WmBhHz7iSOKB3XI2e22QbGM9b.c4y+glqbaovCPei7GGPOm7S8AZjrCDB4nM04pSpXPqYMCP+N31q0Uk.+cS.Bfh+0ynyz5Uc.RRzAHNNF8rm0h4dOyCetq+F5POPgpaoKCu7q9qN75Ovtl46rvEtpNrFt.NmBEA.bQ.bCye9SXL8dfez8c22YWuxqbBcXs6gOxQvK8hOOdm29swoJ0bZYtigyE4snujr5ClqqJq4PeHdxqBB4AYWA1YaW6wRajIquLkS1A0deGpR5mMfF1gCfJdm.YMqRq12QU3SVt3j0gPLrDKSpzAxXzYIYdna3+y.3KADjO+moL1dAIjAct8kLh0y2dd36EEhbh9QOmmSXyIin2oWndrjULa8UEv7cnHPf.k8aQ..LY8mx.RPBpgGoABMnALP7EdvGBSbRSpr7UaAN5QOBdoW8WgO7CV7O74+4+zmpCqgKfyYPQ..WfC228+ne+qYlS+O3tm6cgt00t1gzlM2by3sdqeGd4W7EwQN1Qytv0.LYD5O+jzGZG+yF+jeArAdsA8VMvAgTFbR+30.jcwf4uW6yFPf3zPx3RbZ6eLBqkIeGorgdReYbezZv22IHLNwr5OsCJoACQ+yn32B7uQG6K+ryOKsxpixhexovmM.KC8UPn0OfcpBj1VeDLGZ5BDVpkqDffiekND9z+6WQtaUPsNIz8RgIbkWIl+7eXLfA1wcJetrUrR7huzqc3ku0MN0O9ke4M1g0vEvYcnH.fKPgYLq4MhqXbCdo2yblSsSYxSrCqcW4JVNV3y9rXm6b6P6QJSVtAxJ1euoqcZk848cdT9gh4U8ffeXb340O6J124xts7Bsp98y702KMKqpKqyfiyLkjQOGW5DjIi37YYPfLuy3rU0.bIlMo.6gig9mIwu77OPFz0jKIyVp8BFrgWUHTAOP7uY2HvzWGLfkvjye8zEPP1JCj8bGPy9IsWqKXfxuV.TA.3TWx+ddbPnoRnhJp.y5luEL269tQ25VGyBE7vG4H3EewWAK5Sp667JO6S+U5PZzB3rNTD.vEfv7l+i8UupoLk+Sy6dlKpolNlWn28t2Edlm4owJW9J3q4mIly4uH0xlEWFG0FGbIWnUk0uyacwGv4u+GfFcfDzh3i17W7J3243EQlcoC3UEC1CllmxKn.iBvHud9SgTAgVF+LYMSquAc6GvYq9ZP87sD+6erzVN7yi9gDIl+Sevjxw25neFm14fuIFf.zOuyr.quZa.g.xgWj9HLNt4RH3mHYKmp3qVY.A5x76we1x8qFmqBpxxAVYMNNFcu6cGy69d.bC23M.mqiY8AT2RWFd9W5U125VYcSnXmBbgGTw4ZFn.Z8vDlvD59cbm2+Fu+6+dui65Ntcz4N2oS61rolZBu5q9J3e56+OfctC06u5Lactj4DGjSzPY2JW2O3fj.GbbCWVm+Lcr6WeWEQFibIyQKceUfJorSTpwyje2kL+5osuYaFBIPGhGsYCCt5F78T3.6UM+J.3fhD7cAc9wduf8RB8satPdeqq9coc8hiHk9LG3Sei7mO9sN5Wd9WTQYk+74+VA9wV42O3Mm2N8vx+vbujQRNd7Sx3e0i3nuHiIsPTjrlXP553HI3ExQtd7e9PncViHy5c2hssLGVVJljqDRZf6m7jMgku7kgUrhUfgN7gidV6o+4Gv.GP+wUMkIWcimnz+4d1m9dn0tlUsnS6Fs.NqAs7nxB37B3dtm4+nidbi8m7vOzC35ae6XNAvV4JWAdlm9mhcu6cmbAiQaugFrScc9Mv3kdM...H.jDQAQE44HjLv1JKwep0tfm+9dk32Nm8I3Hm5dI3G6.uu7yPF3IGdNphMxujcYlfVT9rEo2eqM5Kmd3j42Cs0HUZu.xj7XAR+uLUuHL8Cm8bVZUd5a0w5d+r64eSURboY55w+YOS9Cgexev3yTNPEKx.1iv2rOqMiZYcjDm8i5TfENngRgVgeA3GirZ5Sj6Y6yhY9KD+mU9b3ll0Mi4ce2Gpt5S+0NTykJg29cee75u1qu7E9z+qS9ztAKfyJPQ..W..2+i7D+tO+rl0Me621shJp3zunMG7fG.OyS+znt5VR5UDKogVbahiyjKRFWzkTkyPjtVq04OQJmKySpOjdx9g0Iw3K.79B0oaShO8brRNZ7K2tRFIA25PwucsNusPf0I.70YoOYP8tO8U8.J+79EQHiidR9orAQdz2HgH37s6I+4QeciZ4egCBQ+bmNiT8YH70AOXkeK80akNhlDN1UnuU+POWxzFAy4NP1OSyBynWu.AXw185Dfz+knc0PZf.YviCnN4HF17bp.IpolZvC9POLt1qcFsJ9okfFZnArve9yepMT+mcau1qsv2pCoQKfyXPQ..mGCSetycTSc3ickO38e2cYzW9k2gzluy6713m+yVHNViMljIM6qKwHQD4zJE7cLoMNCe7CLO8kCB8r72Cd.dUWqOE9zm1aY9NB.kSYgk8ifQ4LN0YhNXkbpbfp0LNJHEB4rO6VbivK.CzR3CcaXoexeY4e+yjgX0yXAundx3dRZgP7eKQ+P7uIdgfKpf73+rbXKiu2pg2iA7qxhDLfz+aCXR63Nq9iN3gzegCAjcP.sVALADndWo0UU.BxtR+smTfRyj87CPccitIFSbRSFOxi8DnO89z+TC8jm7j3EekeE9v2eQ+fe9B+IeoS6Fr.NiAEA.bdJL248H+mm1zl7eyC9.2aGxJ2cW6Zm3m7i+gn90TOOmf5rW0aeO.ky0rIuAZdrMkVE.tLlKCCZG+5mjc9qNZd0f46QeJQ0GCu1SmtxQekvn7Ln2EClE.owgWpjFH6yfzO0.qQ+j57QGn.09V7ym9Jshv.dNj63frmXgkk9ryME+mIKeO4OPF4oBchCqL36q+sNuK+VoSQKkSy77AmgOg7r7hCzr.9hMGxT72g.8VHTczCqm5q7CBPGjCxHW1JEXCjMupMnCTqDhQ0ctKXd228ia5lukNjCQn5V5xvy+hu719wq9SFMVzhZ7ztAKfNbnH.fyCguvi7Du2cL6Yec2xrN8WstMWpD9M+5eEdwW3EvoN0oLYDnW7awVaFFGHFGlFifgbHD.RMrk2V3Kzp425LlaHu1EYbVl.kAesmCNvAq9HljsXUa5rgLP5nvNaynLU+HbVyYvOny1Kzf74+rxeNAK3CdATYb7ELc8rAklWfGBmkMq4rbuOsT2iBFrLNyK4clBX3TSvD4yC5.B7yvub3yAIjIvkDXjiZT3I+hK.CrC3rCXW6ZW3Y9Y+xRKei0ea+1ewu32cZ2fEPGJTrK.NOBlwrl2Htq67N11S8jO9Hm1UMk1T4zCAaeG6.emuy2Dev6+9luFZ.YcZZVAx70jm02QUnqkAb11VOO+QQNDqJKZDGniKCO4uBt46EK7HLxSh25VK9rCeSl1Z9OjnkFnhJfEibSAT37Blh+oEeqixr76oEvmvfsymo0fePHe9Or7Ga5SY8Wllwku9OVOTvD4kPCH8+RPjIHKAVnG2posZLhhVzyTBIK.UtxTd3WJ0gsyEgX0hDT67Wo.xU+IN+cpfWU5MDGnYzGhPp.s8F+df8ue7tuyaip5TUXDibTrbzdft0stgoM0o3Z9DM+TcoG8ZPqcUK+kZ2MVAzgCWHmdwEUvce+O5e3Tl7D+Gevuv7Ns+pdEGWB+1eyuAO2u7Wjj0OxOacJ6d84cu+b6l7f1LXKGvawp.Y86ePp3Wpe8yaVw+7O7SaK694m0CJ92NMEAJacpQPsqZ8BUyjFuG5TFX1oV.Hj99zFJyGtF9QTSkhAO.YuqG3ihD8b74mPH5zJneq5YZUPnzzS9Yr51NjrH8LauyPk2GRekz9.TesA837pRgpvBpJLXGqXmd.6OCSe+.zo1jgVXMB3OMQwDOonhMHN6IJXbIR9c78hiiwnGyXvS8TeIzuK6xxk1sVXwKoN7buvq9YO8O7eXz.n4S6Fr.NsghJ.bd.LuG7Ie1631m0+kG39mG5RmO89Rdsm8rK789teW71+92hOvTBu8xnrnxlAfN6JBewXa3.Iz.U9bmyk3Pwat0SLHGkglwdNoMzGfypldlL7rWPJzZcPBHwhO7v2WxRZSuTPUYUYvIUWJPGjy+Hm38hZV5VQpEroJK8Xm3bOxEkfdojrqikRdfHWTBJoAjQ3ED+1.8M7a6t5A.1JgnbVmdkr5eMZwb+GkQuohQP2mEma+uDTWVgPuHTk2kzbc53uLU2v99WB9QIe0IissO8eszBDzk9ONPjXhtNtZEBnpBB8ro5I8C5bNr+8sO7NuyuGUWcWwvGwvOspJ4fF3.w3G6n6YW5Vs+Ecq5AtvMsoUum1ciU.cHPQ..magJdrm5OdKO1i7fy75+bW2oUo1..d+26cw24a8svN28txzV7K3tDmurybm3LkMHfrFTECDoWvCX7898jLwROrTTzRiWrp8MN2UVjEdz5rWy2jAuj10i+7wWIWZbMN+yHe97ux.aVMRfqgrNDS+a1Aruiyz+iteTp7Fm5LmBthcTmxajy8XfzoZIxv+rST3j9kzeOO7oyWgxReM+RMPqQNaw.ET8sHq92LlA19OJCdc.Ax3mzVjF6hr3mdAjNfwCeY7VlfKLwsPiS7G+mP0LGEw5.531SjgxplzzOsEzASDd59nm2N8G..kJ0LV4JVA13F2HF+ULAzktzk7oeK.0TSMXpSYRUb3ienuR+5Yu29pWypVbKiUAblBJB.3bDbCye9S3tus4t0Erfmnmi9xG4oUacricL7C9W9mvq8puJJEWRx7HzK7NcV1x0R+sz.G7vOOmYL9xTHvzhx72oNEAQhSCxvIkwutL+QprtHZa40ry4oRnLYsEBecFOze5jKxYkI5NemApnRZIHjSMhWUN34sGlKMqbWRo5cJ8GejFizELI4fVsHJcosunOcoG4wFOQfptBP1uKBkCeRiUN56RCDfFCwAITRx5Lll5AJ.A+UGeqp5A95eNzQiiVm3kWglH+lsdI2WKNEo2k7Io+6Ohede787W6jctBkwu+GnJAujSXP5qDo3O2qTFllWFuKA7SOahNxe65p6eo9N95J720t2E9f2+8w.Fv.P+Gv.BR+VCTUUUgIOoqDkbQ2SW5bOl7ZV8xe11ciU.mVPQ..mCfG39l+e70dUS6UWvS9nQ8tWmdGGmqasqEeyuwWCe5F2HToKKkazOaWsAhLYNSklLaPBg.+r9MtEoLDA4bPLroqDfw.KwmPl2Ul8UFhr+TYnOVtVd367wW6jP7AJLiwge.OAkC7xrMJJsj6NWpCdcUQRblxeCCJEy5G1Isyo3hTm098oLukN+0NNmNkQemjMOjJ.ncLVN7yi9kT2iWraPVyGzNZI1S9ifZpIzN78qLP6Hf..G+t.0+SuivNqI4OS+u90GkqUBM1YuKW7UZOY7mh9R0Krbu9cFJ3IppJ9iiKWEAbP1gO5pUv8m7.JmYMAnCnhVpCQNGN4IOI93O5CwgO7gv3F23a2GLYQNGFynubLfAMfqnhN08+nk+IK9afSiIKp.ZePQ..mkg669ezu+Me625e4Cbe2K5TmZ+mk+MWpDdwm+WhezO5eEGuwjsXKk8kCfO07Lu3SF4SmBfjqo9+p4kubN+4xP5kIQEoF4iiigqhH1vdEJGtRFmdsGDCh.vrBpskb2ydmS3UWjikqHN3FqgxXUamAeE8sNTZcN70k2lOe3or5osakZGOX+VF.fnHQ2DklWtR+gTmwReTZFcpqKhpb14SzS6uflS7DcEc8VK9goOUwGmQG6RNLmR+8Rrd1J+I8KxzHX0ejSHR25ZooDG59rrieReSH19ntnPAGqw2a8F3EWXF7Yw2wObF70brJfZ8XToBaI5DR9sUcvxm9sqNfd54za4RM+mHyQF7AieBBa5y9Lrz5VBFyXFK5QO5QP51Zf9eYWFF2XGcOprK09mezA1u+48Ve8Gtc2XEPaFJB.3rHL+GcA08Edv6eN2xrtwSq46+.G3.369s+V3C+vO..VCyrAlX8K95W9i7JWuSYrqLasOcl9dk6mb7wlYSCvfNfd3yleXMLpKwHkUj9YLFtMzWBLPvqUfebBeP9RzN7sxbqruwadsiA3ETGqqAXmaBOIYTKNXDGNwrNxoJML0Nz7uCtLxzWoNGjsgFMuuB8EcBclKPSCiNCzVC94Q+X.0wxrZ5dT+dhLYkepRQB8i.EnpdZFH9nj2zEz1BHvp+gm7K8QZGkxuyG5TL+mMC9fNjMWSulDnaQOq+wpMcszuCFtHV9MAmPQNkqza6+nemF8qd8g4EyzDnC9NoLV3HG4H38du2E0VasXXCa34R6VBpolZvTlzDprwcrm+S8oW898W6ZWyFZ2MVAzlfh..NK.SXBSn6y89m+tdpm7wFwjl3DNsZqUspUhu4W+uE6XG6..x68jQDsiLcf.lrrcDhkGeMXw2wFck8wOEDf3fO0NA+o2UZLIaBe9kbzwFlXiShQJl+oGIVZSNKEvOXV7E2RLkZIv2Ii+4X.bNVNox2izL9zNnIll9a8Bhibn5n46UKuPBlib5RNxiSmRcJaZp76Z5KDWbDS5e94ak3Gh959YN.gTLz+tuC2XS.AHL8U5RNfAmm9OPeT9f0wFOcQJ9mdLNyY0vMwocpLnltiP9g4wk59.GX8Oe5Fl13j7qmREmS98jfuRqPBMs.v9ddYkdmXafdOPGHPP0kGuSzoToR3S9jkh8r6cgwOgqDUVYksH8CActycFScJSxcjFO9S1sd0mlV6pVwuuc0PEPaBJB.3LLLqY8.iel2vL27WdAOY0CYvCpc2NMWpDdoW74wO8G+iwIO4I.kAirf3rN3MN5S80oKwKXiENC9ZfK0epUMSoPSKSKyGoYqRYURKdLSVDJCGbFPlqqJItVNH7YqjJYkEwTCYrSeGSCC9F4Lf0N0pRmBpAwfC1wVAEIqdh9INm.+SvYA6sHIc1E.eRl4wrSTmh+AfYc.nmSYt+SE4ju9KsGiuFqMT8EsE7cF7cF4WjWUe.w+b+jZXU5ePe9lkJJX0e7GgGlmgU+6j0ZPbZiq6+JGHiPTUlf0+dOmg9524D42bMFeIHCt8zOKryuOT7frXAc7XFp+mp5h+1vTneddzgh+nmKNS+CE9hrNJfvmJ9eqacqXo0sDL1wMNTSMsuoDnhJp.S7Jm.pphp97QctaSs9Us7moc0PEPqFJB.3LHLm4L+4NsYNk28K8TOdTsmFyS1gO7gw28u6ai2+8eOSFA7667j3l75pd99.BkYfxfUvP9sfy4fqhHigzDm+wRIiUUQn7z2OaHX4ERTbZtSb1J2mvAhQN13ouT463urBKUyWduySN9oLxjJcDyKVKM+6TsCGfhAeICZnvmzY11DryLRTzsOGziU8AOzSzerB0loZ.0eKhub+3xz.futo+OsMH4mqlP.5S5a85fPejQS5JYpKR62T8e9mEE4CVdNcVxyD3lDHYZo3ScDqCNH48f3LUfSzmNs+ZtczOCq+gS1I.z63NlBIUCvaZQb.xVJL2.AD9mqFfHllJgHAnHRXrFe.bzzoDnO8t2XHCYn4ojaQXTibD3x5W+FOprKO3pV9R+ts6Fp.ZQnH.fyPvce+y+O85t9Y7u93Ox7cc9z3v8YSe1mgu9W6uAasgsZh3lbjFEItZCO+hVGw1qkuSQNKH5Yik8xu9X6kbLQ6keaVrZ6N1LFXIfs56r9QHmrZCuF421lViTjA5V1welEtGc.3vxfZWLHgiXjey7nSYpozA7VjyCempOQbLp18BAbbXjexwrm7CE+nCjRpXRVcZaEemh9rTw8epre0NdTApYjepWx4LUAfG+PWyQ9yyp+oEbZLrSGiKMP.8YVPKuyBTiuTiezUbh6yhTxBDGob1yLupmaeYbfs5cp9WMsTzmZSpEzu+QABnsGHu+FaaLO4izUIAOnnC0mp46L7un1J0byXoKsNbricLL9q3JB9Q8p0.Cb.8GiX3C8xPUc6e+xp6i+pHPuTAb5CEA.bF.tmG3w+t2wseq+eee26ba2aSF.f26ceW7896+6vwNl8CoUnr28cJl4SbK.m8PBDN6+fyinW1GripTmi77DqvW1e09NbRXD95YRh2ZThxfJK8SaOndFidHefmq3HYg6QqTch9kRInHaraPl9gV3ahNN4hToqkr5E8mzPptBmjctVOx8Oz+OC5NtOQSe5d5poP9onmq8ieJ8CH+73Oe+NNAON5BcqGYoOkYeLnwYhVPq+4Q1NG2eQKLS8gZDceZ9yaaGYwxX.SgOz8Lz3eDveKeMmFkDckD6PZ6lPqLm4EoOCe7FqpR.095cKP5I7aqTtTAB3QGCwMxBjyU.hWcNrwOciX8qcc3Jm3jP6MAnd26diwM1wz0J6bM+40z0N+2uoMsoi0tZnBHWnH.fNXXdy+I9sO3CLu4eq2xM0pbFEBZt4Sgm4e6mhm+EdN.5n8z46bVxPmbpBXiNO8BYxzfv2.Dddk5WxFNAe43JkxDgxTmZlvYIHFVzWmLXnMfEwFA4fDbR6HNJ8cnVFcs+p0mVzXo7OkwI.Ude1UOymz7uRFc4rTgdezK5wLxIKujQa45F4WYHl0iZ4mcTSJAs9mFKjk9voze5yM+SG7Y5qz+Nyew7I0WqF9PjW0GBUerReDP980+zrfYpHSJ9b+qiNWJnpD.NHqV2NJfjpXEe6Dm1D+SzEjCUk9iz.pwDxnM03Hdb.I+JcipT7tT4kdFZ7arCR0.zA2FzljK6uyzgdGyhqQ9Unp066Ye6AK9i9HL5QOZzyd0q7TpkEpo6cGS5JmPk6ZuG5Oqyca.uvl13p1Q6pgJffPQ..cfv7erErxm7Idjqc5Saps613vG9v3a8M+FXo0UGazyLWsjAQ+zp.cM4gMYipdNevk5URmQt9ni0jgPJeHeg8.mITFiANgmjrLQF92o3eJgPYJ.Lhj8ddxePP+IW043soGU5b8Igm7MbO87vO0whrJqkLhH9W9fEQojkfu1wft+yGe+rus5Js7S8Ao202Xrk7pfFsykqoD85ms8fuCpTDKSCvxjHuF42IA6Y1a5B54J+xGZmzsHHh4JxvAFP8IPlhJJvEtBCd6nfV9PGhz+zGLHE+S8IozgYeUGFK+lpmn0IR.iopDnqnRlwez6eJYM44re0Ak1N+2YjE4phmzKBXy32XO9W9bH6.vwOwwwhVz6gd06digNz125BnKcoyXxS5Jc6a+64OoO8reua80uxM1tZnBHCTD.PGD7Deo+js9EexGaTSX7iqc2FacqMfu1e6WEMr0F7LvKYOXJQqxwotx.blAxS.wfkBbhgPmyYL5Qk2OICJJ6WuJIv9nbFiCD99auPxfh10OfxPm2yYkevxUqAzmE8IN7cJ9z3Jha6nTmFNHq1ZJ4PV+GYkeSFxr7m0AMy+Q4SevsijQL6DTYPVGHggGDxaBPhnOI+Zp2QgOw9h7q6hof4zANP82p662lJ56K+zfad7hBepOT2+o2UAvAyzFv2Cht0rnAK6ZFPw24o+H8Vn.uH8B2YK5.Phoy6dPB3vouuoxDp0dRJck.bUUWS8SQe63mwxmzHV8TOnumOelRqRw3SV5RQSM0DF+3Geq9cXMTYkUhIOwI312AOvWra8d.qecqZYKuM2HEPFnH.fSenSK3O9e2depu3i2uKejinc2HKe4KCe6u4WGG4vGQLljZrJ6IWG.ag0Iuj6+LbVjA.SV+P1W6I+NsvxRZAyo5lAeht9ms3IOqDzhfpSie5yJFiHq5p11XLoLFN7LRGCpbvIlrJYvltdLu..YCl5LxTNnDfx.UXWhW4DkyEeOYTk0L2dJwLyz4n.I.OU6HbG3r.yv+Z7ytNQZ63K3jQ9Y92S74mQGDK8PwsR5CO8eYvWEL.cFKnoOEDfCpc0B.34Rm1y8DJs.3+9qk8i46Eq3eVfziEbJ4WTULM7+bGCHa2R4DWL4kO5rpf0+LIbv+aCfQN7bly2yylPB+nk+rK5wMrg0gszvVvjl7TZWmW.UTQE3JmvUfibzi7.co5ddj0U+Jd+1biT.FnH.fSCXTiZT09Edju3d+RK3I69PFT6eO9+F+5WG+vev+BZt4lSMX3ruXptl9i0C.YjQrBneoODPOuInA0gZCcbByGHMoNy4H6oLvXG6DuBNCMJKqD5k9S5eL+KFWbpGlxliCrnkb9SSLO+4qkoT5ATDsrwnJY3Xz.jJaPqILG0dZOKlRR6+kTz5LUjeK9blmL9hNzW+Q+KNiyLc+KDmCT+Oce1CmO8E8OyCmt36wqr7a3eRtrie3gUB4yk9YkeO8e4v2W+wio0SSfJvLGEzXr8DPzIGqy4OEA5woV4m3O4kCh+gTQjHg9rdWURcSY34wPjy2zw2Z7ItxIxeL8dqpZeT.NYpFft+y4ce8qJv9EGjkeu.L14N1IV9xWFl3jlL5Z0UGRAVVHJJBSX7iEkJ07rqra0100tpk+Fs4Fo.XnH.f1ILiYLi9ei25b1wevW9I6zk0291tZilKUB+ze7OBu5q8JJGcJGJFG51T.7eYLumyGzYko+FvSGpO.P8IgkL5qwWU75TiR9GFQFm+rCA00H902XBj1tkjCP7OfLu9vtG5MAyj1Z7uGP+YXjTl0k5j1jbtYdXIZYkEICNmXf2EBee5KJPFWkAUiVhttpZILDh9RSqbFcZhux3uG66w+xZafjByzIn98fzOi7mU+mG9vCeSowgZc.3PhSQd7hbnBoEe56RPKe1BnTV5r6oqxAJk0wKUNeq7aTtYbzZecIjC4z6ve4Ic72T.ShD4.92gRFQpDH8ThbSAaEqZ2CcvChO9C+.L1wMdzyd11+Xn4bNL1wLZTUEUd8UVQ0CcMqYEuPatQJ..TD.P6BF28duC55m7U2vezW5IqrWsiAv..G+Dm.+ieuuKV7G+QpLSboFeTuzlZ6vuR.Zmv.dNvB.lL+ijWZICLTlAjIU+uUAB+o979l9xclRc5HmvNO7QZ.EZiX1mmb8kmbn229z9sOVgO6jwIgTIN78zsJFv5n0ZnVqqXmJlfczYVpz2YvGJ7aI56z95T5N0XARiQxkm9NC88j+Nb7MxOT7er0YtzoyiIEDxi9dxum9m+YaFeYLmCNuJeQxnTAA97DHUzHmn5wkgAGSWsyZmh+yp+STf5sBH+NmmbxsSr85HyzZnFKBfjC3ojJZ3z7dl.RjwHgBbGdW2FzgWvHNfSdhShOXQuOF5PGFtr92+7TZkEF0HGA5b2pdZQQUOl0r5k+KZWMxk3PQ..sQXFyZdi3lmzU9Y+AK3IqnG8nl1UabnCcP7M+FeMrt0VOHi+zK4YeQM7KVVmngcVxN8UFroUEOMO+kHhoJANaTW8eBwfWD+B9Zdltr51Jm6p1jaKNuPoc0mUt5xUlx+w.putdI3qKwul+0mq51TwnxTJ5RguCavyOiWs7oKwbV7cL99AanoOGnFbL+ax.EhK.+JlHF28ousE5XwOj7q4eqrppLu7GoM.mUom9Oi76M7q73GtBN.5+Hj7SSQfh+SwwAmL9S+8fvopJPtSSfZpsrE0H88Og9Pq+UAIP5Ty4Af48O868BAHQgBPlO9joOvP5we4XWgpNopGvn+3q5bLOw3oBRoToR3i+nOD01y1+GSnQLrggt08tN43Jpdh0upksv1UibILTD.Pa.tsa61FyUO8ostE7EernZpo6sq1X66XG3q+U+avN191kWPLFY0NYA+FaHCoZmsAAm1zN3xjSKLNB+nTi1R4yI7AanSx.zOHDI6W1Yhxwu1HD+YHlZGemi9.47mmee46IuKMBE899N4dvyGubX7noEyar92hiUVRyfUEQSFm8PJsswkXqE+rc+baYq1ijwLmknyilsD8g1vbGA9Yk+P+tInU5uCQdtyWneF42S+UN78qVVP5qbRlLlQ9.+TJ815ECKcxWpWSNzINH+xPYpF.T7DO9ybe8zb.uwOPcOg95nb7OldAb7NPQOEeTaDqvW2TYizR5Ax3fWclknz5xGKJisLvAtr7ksLzbymBie7WQdJrxBCanCE0VaOlvohq5pqe0K+oaWMxknPQ..sR3Fl+7mvUcESY0e4m5wi5d25V6pM13F2.95e0uJNzgODeM1.F.n4r1Aj9IkUbTommPSIL8Ayyj7TUPewvROO5iT3pcdFwd3gjIDYjKNKePDHC9lmwY463xv+pLl3swmywmjajwXh2HWKlum8oWMRtn2TVH+Ter2B.0B.TYr2o0kZ92upFw7ZlHgFQsQ7g46Iejym+kLUYcqd5AH4m+cO5GmU96Xw2J+Y4+jeWO9wjUNQejm9Kq7mspRgwWq+Lz2u+OD+CoBDxykR6XgCHZxi+T6bfxe3BojBm26JJ8hTnjr8+76bz3m3r7Ow2z4FPrBeIffnDYMNVllPOdwGz5+HdALpz+J74zQz5+TXsqecXe6cuXRSdJYl9wVCLjAOHzqd2qw0bTmu90rxO4G0lafKQgh..ZEvrl0CL9oekieke4E7Dtt10t1tZiUspUhu827afSbhSjbAsiAmmgEiiR5wcLN5UWcHvTxe.d0KysSJsXCltz6vueliQHCukC9FZDvIZnWt4OwYNNqexvCkHEGzhyY9dxa+d1m5ff3IVgZcxoyHTLv4HzMOqQ+yFe0YDmRakQ4VE9vCeVOFqLhp4eQ+Yxn1H+pwOH6zTblDeYvlm7KOrL1xriU.KuvSGk23OJvPq9KG7Cn+X5KiTMA0H9N86+Rvm9LHSi+ncTB0.RecRP.Y99CX.m4mjCdaf1ABhltlS3Ml9YdWKaaR3SeGEbposqTM...B.IQTPTY9OMf6JbdKRPOchluMAkDx9gxtShdvJSN.zPCaAMzvVvTm5U0tNB0G7fFH5Sup8xaBUNy5W0x+Is4F3RPnH.fV.l9bm6nlwTmzZ9xO0iG0dc9u3E+w3e7682iScpSwWKjibcFLzem9ahwR37dgGlmWafl.87hBny5mSqP4HvynOYzV+9OaPM.9T6pCHnkhnmx52I6JA9HMExGFlTwmcXQFh442mkSc4iI6QJcobw.xuNwJswRu1PK9AvWbnEFejC95NNqeHoEE9mlK5XkiT0eqnuVAdl.esblMydkX4H5.wYQrd7RNzWeEp8z5uxgOgTreCjbSq7G1OsdHL01z3O+SpyRJ5SmI+N0yUNPyArLXe8E5tCWJM3qokmLbrR7SUfI3CSfLkRoOUkfV3sWX5wCo.M7uy9CmdKHlrMAW+5VGtpoc0sqyJfAMnAhZ6YOG8obUU74DtU.EA.TFX1yd1CcZS7p13W9odrnt0NK6+676ea7C+A+y7QLpIpWJBdNaF+6Ci0yVJqe9H7Dfmybc41oEzjNCXvjH0ftlOxDPBfXLQmAsO9NuelC3UxeZe6SsWrNKQCcR+chFjEMJ6MFOg9TVL5r6zw.vN9HYBJ8ul9ZbBn+XmatxiOBfOuW5g3zkj2X00nFvuuBl.I7oerRlkwcxOaK3mk9lJijG+q6+H9VoqLetY8nOGrrKK9L6GB+bnuuSTQ9U8+jLJhV.42wiE3yxAPaePoT7zJw20plZ.0n.+fkLxOo9n.2z1HT5eyKPNkr3r3q3es7GGG3yhrG+psKPpSR+6f7dN0OAe56j0Jv916dvpW0pvTm1UgN2o19GRngL3AgdTSOFOhp7JVypVwOqM2.WBAEA.jCLiYLi9O4oMyM+kepmrht2812B96W8ZuJdlm8ey33IyKQ9Fk7uFjWPyCLk2i9nlnnAe.3DqZKic8bbpnJUs4YT3SN6RQRL..A+ff5L5OJx98bW976JF5nRuJF9DAvTpWsgHkUmH0yPxDm2RprlwXoxnkoxJJ4SWxbbZhuoOlwOV3eIBGwPJoO7Hum3ijs6Eit91hNqUfun+BP+fxud5RLTiI.67R2ulg9duin0ekEes9S3cBGe8mQ+aprg77ZEXInK4uk9z2UBxiHOhMkeZce2Aj2As8wAN4EMieD420pvWBd2R6j9d5ahgndx2dDS9bddIfiX9g0AeC3vAO3AwmrrkhoN0ohpaGGXPCcHCFcsqcchwQcdT0u5k+bs4F3RDnH.f.vPl8r68sL0Yr8+fuzSVY6cq98Ru3KfW7EddqyckQLNJYOG8lrycsvKZJCqNnNNeUmlXw78cYLh5fKcguoou7g.gLVxlLLYLjx6l1z2QVN7Ns.8hAe.D4.3iiU1fqScrlBktRSS8hECR4OYiYFCOPcO6uyxpNKPm3GyhmfiuiEAemG9tVG9lpHDxvomyPO8er52yP+HM8cYzEvUN7aEz2S+kY7qJPQQl8oebP5aQOq9Ke7ym9YjE3r7uUnBJ+xBeM4YKkRe57ovve7+O4mzZywAHmk.YBDHm9eRFUeaJRzUxmzYGMHVGDgQ9sAlAGD9WecNP.pJAR6vUOxvwNQeYT+hdiXCsSe.aPkG8HGEKstkhoL0qBsmoecXCannSU2koDWUWFzZW0xeo1bCbI.TD.PVnSO7cbu64O7K+Dcpm019Njedte4u.u7q7RJGlIWWxzx4k4s04htb9gxh1e5BHPLnHNvnpAvYpnbTXC9fotXz248bz65NUEGhs36yqYfzOuvz9mF.lL9oOLKztUHl4esgEMMgxnTrXPx+aTNImLhwddWj1TxnTcvBEf9rnFCCOwUevn+X2nFG8D9RLANC8xzOAjoc0+tXTuUR+X6u6WNeKeH7XX56o+Bv+voaGzpouQ9UiG8UfR0xRwW2+ax.VnoQ+Q7O08vkVPMPwe3iWlt59B84zOgebJdz5juTZP65uZegAoOj+0XxoOy.R+K+bx.UG+lln+zhnN6bd5ybxhwEtzpWDq69xmuY6DYzOBCn2LE9SGPiG6nnt5pCSdxSAsmogcjCeXnBWzUWQUcqh5W8x+cs4F3hbnH..O3o9i9J6+O7K8Dcsu8o8c799y+YOK90u9ux5zjdoyIFMzY3KNO8bjA45Lj5bidVJqgnnH1Ap1FAsG+oWrrNJT7gx.NTTVmIF4.D70cY3+P.cJoQmS+xYrtiOM+H5ZyXRzOYxHj3GJCBG4bS+2zkcL+J5Oo8rxul91Lo7P21uJVvL8a4hOyqd5eqKNVV74eBOnZCM+25nuV+nz2jbD6ieKSe3JC+yi9T5u.zOH+SxO0BAduhw2+8IlVJ5qkenvQMtgbLJzH84hkx8Wd5qWSLPnWr7tDuVA.sfAiL6IeKP5FaUN3W3S4nX9YsUCPzs93q0WReCPrYWBP8ekJIKxPAc9273Xock3vD6W7vHM8gnm.b3DGuQr3krDLoIOYzdlN1KeTiDMdhSLqJqta6dcqYkeTatAtHFJB.PAO4B9S1wWdAOduG3.GXaF233R3Yel+M7a+MIeaJLezX7fLkFSYrJuOJGYvuhj4jkJcHOG5oQlGodwWJqleI9CPe80RetXumiMF5m9bHfxrIM6BZg9QqxeGjShOmKI3.c11JK1rrHYIBXydTmPoikAIMMw5BmcCSK0ZIfZLiCSOmBL9deQ8Rw20Nv2nS0dLJC+yERHV0+QWy42mFh95qmVBcJqYO9W2.4Qeq9ym+E8cKSe3Qe0SZ5+hKK9x3mxPeU2mVWSien1Lj7S8eAoOGXQLOMVxJtmbXa21fkPL+ssH+sOH4JFlJLX6Sj9OfXSV1B+SW268e9YSniNA+D6Lx4E.S2b9hBRfVDH8OqWgXWR2WqqFvINdiXIe7GiqbhSB0TSaeZYGyXtbbjCe34VcupckqaUqZUs4F3hTnH.fT3gWvWZMO0i+3CeDiX3sK7el+smFu0a96.GOK6TxoLl434gEfbRoiDVFza.0Kn5Wx4rEbN0bNpL9.vFAH7hbRVPNlWEb3+IOjhuTNVoFOGm+rALp5DDOktlCnyac4TIL.8UhOeeZMKPFob1mWmUoXHI84X92XhW4sgDGO4WSemRlcV5Ipcaenn+8yh0J6vI7hO8yk+07qPdtOxozpYnuR9om1p+r5edbPYned5OpQjrnU7oG807uz8Xk+1C9F5qk+73e0nDIS5T4W4jW+9ltqTy.r9W8QqJJc7ejAe4rsHl1D9kYQBpqvPrl1NENDuqGWFS7t58Y83FS+un6EyWx4bPjmMsf7oVOpdWMDsHdgNYAI86IO4IQcKYwXhSpsGDPjygqX7iC6YW66gpty89MW+5W0lZSMvEoPQ...X9OxB98O9C+PSabiarsK7W3y9L3M+c+V4Brsb6.bqQIiEGNhbOSGV7cxbwFEEgRkJwKhH58ciYMJHjHmW6AyK8Ynu5ZNU6XMr4wmZiToek9RxzGdFnkoef9ppxLE.ToOIAFYzUJm5pjtylQuVuojciNUz27epdd1WQYnOX9H7px1YZbOuh1jZsNL8ZGeMNWMGSCDnhFYjUMQjqYpnhR+Yk+Vl995OM+yMe.Uhl9L+ixQ+VG94Q+.Mf4VxXJuJy.4dw98ePq+CO9iZGBeZ00SOCcR8Ybvp15fgAkcCJidcU.LibTYsqtVHmv97eIs9CxT4URUwh3rK5AONM1zeQURvTQEE+q0m..m7jmDKotkfoL4oht0FmNfnnHbEiebXa6XGOUUUzmmYSaZ06oM0.WDBWxG.vWX9O9+x88.y6dm1UMk1E9+xe4OGuwu90gtzjTz9.Rzq5A7gxNLuS2O9kfTbpHc9zYmnN4CPRjyYlye8wFJEIMupgM+PO0.R1BRVpvStB7BdpggnHAexXlCPsuniT5D.wYnU+w7jDMR5iplZEGr7JbhL3DagoTCRFPrmcexmZzL4YX8drl9NezM8eL9dQn3ZQ7yg9ZGzNK+yNG80eNO9F4PeHWy+igj3eTK+sN5mG+yiingOr7ireLl70ep9OdzWq.e+fIX4KPCn6+bf1QDZ8mR986+7oO+NkL7k0+FaCB9x2WfD7k0JS5yndW2Bjznd+QMtOS+uh+49eDazO.f4eRO3fiqfQxZCHoAbNm4bNgZmPf7NnHDTP.B+o9oh+ogVm33m.Kst5vTlZaegAVQEUfIbEiy0v127+a6YWa+ar+8u+SzlZfKxfnV9Qt3Etq66w9Kt0O+srfYdsSucg+K7BOGd8W60.fxfqtVcZq7lWZ0QgWdm+byD.3nl0QPyzwtP+Bk+fMPExTol+0N8CzBpJKDE4.RWKBTsHnE2G+cVO8dzynKGotzx97OeCR+F6gO4nvI3xNeXGRx005OwOmpsT5GlGzHFK3y7pRV3epLFGn62hO7vmnO6fRI5vCeSFvFxyxPv9ei7a6kM3qnSKR+xv+51VieH56q+xz3kA+XM9bv.4S+v7urh4Y8GKC9YQq1C6J4TO9ym1B9NF+Hj7dRoz6mLsdI+WI.fRw78Rdf7rLniYiduH1W8I8+wVdguOjfgzMbIFe4hQQQY3GIXbe9TBVmiEgniV+6bF9m6aQLNvA1O95+seUru8t2b0A4AcsqcEO4S9nUbyy9t2VaF4KxfKYq.v8bOO7C74twY7OLu6YtsK7e0W4kwK+hunc.rI5dN1UnGLqMH4GkqFjxekBp81u4qpkxIusjcpn7UufyOKQYE+qh21D7geY3XHFbo+0eHTRxPPvfxHvkxb1.OPZLRpqIBgQeP5N9YIe6jiBE99Nz0Y1Yz65LVbBGqSTRGfBzFzXbT8sJ9UqmZY7sxru8X5Qs7OICnLxuz+kg9J4WGnTX7akzmGukk+yHmNO5mC+KsmW1k4fuQ+WN5q3YafF5mQBlz4+b9xBI+Z1gBVU+OFGhkc7OkOLORfywpfo4.JIug4r9.LA8Ybv6Se4cg7F+JmBiIONsNhzOGUQ.+opPNreLJWCeZtszvfCXOl4Pi9qwFaDK6SVJl1UOczktzkrJgx.cs5pwnFwvq5Tnx+3kszE+21lP9hH3Rx..l0sO+odMWyj+sOxC8EROHZZava8l+N7y+YxmdZmSYj0IFT7czGz.kGHNpjmNhlScj3XUmwRIUa4.Lk3WJGn7BuJmOOdwVBYwkgWF.DPFdhkcgfNiAZwHZ7qRJKcFyNCmX97p1h3GK3az+JiUjLIYkQ+TLvCcaobjaM3nIle0KzNIU5OeGVJ7gO94Qems8Iok3eI3LUVUp9eiA+PzWy.NQ+mEe8X3xSesNVy+beseF2JAT+cqPq+hMTOe7sA2R3qy52J+l0w.0tNsuQc+exeyzW2npp5n3.kdTCYG+Dj+A3oEP+NBsR7aoEInd7q0ubrg+SBVv6.zhjEVQHa8O8NGvbdIjhCuvj01+xYMLvSyfhW7SVHa0sD82QO5QwJWwJv0bsWC5Tm5TPZjGzidTCFv.FPMG+TXNqdkK662lP9hD3Rt..F0nFUsy5VtoM9EexGy0413.F.fO7i9P7S9Q+q.Parv5jT6vPxjmdYRLrDDbVWzIK1O5LxO4kV9i4i5EDsSP5Ozu3PYOj9WVmMddYBEj..rmc+NGb54yW4XHloex83C5G1ylXXWBTRgjG9wkEev5g.QD.WF78bvYL1SZGU0W7BNxPepOMiFyXQu0gOcKnk4jmMi7abr3nGLi9SSdiiHkiQnzgRTLs.9ki9oMPY4eRmmJ2APOCCn6+DtH.959LCuDd7inzcgDeUSI3S+e58G40YYDfzVzZIvS+A03Wl+ErnOnPN3324ydFYDw6S+x+gFhVM8FlU3ek7aH.8GjCeS+sp+WM9ksATR9JHlcq6pacaaJshnD0qOIE2y+8QNxQvZW6ZwzulqsM+ADpu8sOnG0V6faNtxKIOxfujK.f66gdh89Grfmnysm8R5xWwxw2+e36A8LI6x7Bk76zOi7ul+nctsj6TA+hSxK5Bsrmy+jSN55LMn4iK8EeISMwGgmPnxdwy4u2mqW9D7qT5d20kXliLnDa3Q6TVnCLwDvhg9Z9ukvWtgI6YktVWgCS1WhJLC80eC6o.u76+rN7oiEVaes1vuM2tr36T3qouQ9U+Tx5Ry+9xe4oeV4ub32Jnu+XTm86XfN6c+PLCQeQ+ks+KH9lwHwgouu7q4Y+9O524.R8Vw5gnOri+cHr9K6WsvbvOG5yyEOfbTBKu.ppJfzVR6Pbf1tfhW0lXhr3Yp9B0Ro57jjShAhh.8sDlrkYzmA.qSdGGWPh8KuEDLhMs49Nv9wl+rOCSe5WSatptCYvCBkhadJtNU8kbGTPWRE.vi7E+CV8BdxGaPCb.CnMi6FV+5v24a+sPolaN4BrgorNczCbIfBTHjyexvOemHJCEZfbxKFxGLGg9hCzTmujwOm5EVsQHiQPG21dlnrLHYPA1ikWR9o8yrCxQ3qRns7hxXroBJZCNhBLS1m9UWg0+NupfjQ+6K+pdiz1AdFXy1+QssNidg9jtJy3AEd4hOQee70IBaUI4q+Bw+9zW2u60+mO9sL8UUG1i+UsYrU9aUzW2+YxH2herO95T40AFP89J9Wx5za7qmLY+Z1Ye2UitNRIy6eYvGkA+3vxeJcindGG3EJn9Csk.1w+4Q+zVKW8OOFkd+iGaJC.hctzy4ij0Df9.MhtlOvAkoldM4Zd7TLIyTELRt6d1ydvN1w1wTm1UaSRpU.iZTiD6ae6atcqK09ZqacqdqsIju.FtjI.f6Y9O9S+HO38eyWw3GWaF2ssssguwW6uEm7DxNFISVjN+WX.zVzyKqeS6olW8XENriBPso0PFcYSYNYdPYfN8Ec13MHCBg2EBLDkXnwThTGEjhT4RIPC9Bd5BHsfxYWFChmF3mTKB+urdYvVLtpcH4YITvgrxQ22kAceGI9FTMN4yEe1OkPemleYtg4eQdoGV0fLCnjdciFGm84aI7KG8Y8e.92q8bPje8QyrPeq9yzJTaFBei9PXemzXlFf3ehNlWqzYPahvJL88keiD3rH6fq73a5BriKoC7KxNAO+8Hc52Zguo.z6oTf.wzVfziNZ4kGdPWh0eptKkbY4Um84fX+ICo7BPHzzGXs0pOQBA1912NNzgNDl7jaaaq6HmCiariAeVCM7k2Z2q9aenMrgFaSMvEnvkDaCv699m+e5c94ukGYZW0Tay3dfCb.7s+VeczXiIiGbdFBXGWN8bckbyjKS6D9r.m4O0lkR1NMQQQxp90EgRf9JckWldjQxXwOUZaSQByQQm9.lLkBweQIemAzevdnskD.3SxOh9zgWh1AmSQi3xx+NwvQpS.RlZa3GyFgREeN6.tx.TCDK3KFdU6WbxfL+7wrLAp8Bn+YmaNJCEk92y9ZV7AiOSeHz2H7j0apcTxk4476aM9FTcf56WN7KG805eO9WzsZC4JYVPm53r5OJnNW4wmcB48tEIVgnuT5aYLliYiz+PlONC9wJ5qiakjUBeopEHC8YBZBDzhul9zVpkPm1O9.I6VfH86tkYqBxi0Ylx9Mdf6CLJPe8mSzSwx00YmC33O3QQoqN3XCc83KEtzTARKVPpe1dFmjPOp++c98uEdsW8kyUtyC5Rm6LdhGc9Q29HG2layHeAJbQeE.lyb9By3Zm40tv6adyM2Ab4AG+3GGesu1WE6dm6R4ryFMJzC9YCLd2OjQT1hmb2nzUvu7MFO85jibsMHuW3LkEO19xk7hj3ExjodHHN8EMWxwLbBeJYJ6bIAEvNVY5qDOt0E9mTODNhg.qtvvUtVI94QetOibhl0gQ1pFn0u.NWjp8D9ym9h.3iuyC+v7uEeskdRvxncxBZkntaVeccWe1RVTd7yi9JYlcZkdcuvv3FltRl3NH9Bh9yNFIL9FVjRk17NKM9WwphGN.GLsIqFbQdcQRK3fpr9ZcgoxBtf3yZjLuJZueH7KAnBDWZ.56NPozuAG7VxISgAD9SoHTUlPZ2PzWZtjGlFlR1un.5hcHw9S5BCjGVEbKBJsobeQG6brF0be4y.cxEpeMqA8qe8CCYHCMPamOTc0UigMrg1oi2T7irxkuzuSaB4K.gKtC.Xlyr56bpSa8O4S7nQUUYUsITat4lwe+286fOcCaH4BdFgR8IJNl.PFmtrgOOvIQd6.L6we9zyyDAs1Hn5kY5EqTFjJelTEB4EFnnWNtYkofHxwmXepVmy9Ws9BkLT7cpRxtSdX5Q0zW1SwNOmjJYUqSCfuH+RhoF560+3L7jXNNiSFqp1Cemk9jNmxzCT+GrFw80YP9ckfpxD2YtkA7uNaYU8yPOuKv0ZK3mGt5.E74esL5K+7s8VeFd9sxjwrR+qwmbXn6e0QXnCR2uJBI9Qn2eRczP7j+3GUep1IkMHEQQ1ZvmG+XnOsE8XEgz5di+Q5yReiAbNG+w3J61ETzC5gY5L68e+KTxEPyW95OVmlxKNYKLJ1HCCNUeB82wwZ8HT36D8e58V9xVFF0ke4nu8se4PgvPOqsVz8ZpouMUppAr1Uu71doDt.BtnN.fmb1261+xeomn601idzlw8G8u9CvRqqN.colR++49hf52s6uX6PbiiNj7HQzB9yQG2m1Rbk8iOhOMEdS7ejd9A3ajIOfb9izETjSUhyT4QNlR8ZKxNOSiXwff8wfXUPLNKrHY0ONid1ZfJe7cTjH51QwifMpSsntAr4pZeNkgw7num76WgD3oyxYHh85gbDqute632ldssSe8XueuUfeP5Gmy88a2P7sYrhyL9vp+snXdAx37IAesyaMxhixjmy+YngOP8bZVleZVVzeQ+.6HRONw162R3Wd56zxLh4OrPz6rv4RN.gRc9GEoNy..BdtAvtQcV91WOS1YHbjuThR+m1dkYgKqnicd9y+KIHYGiw27bTaoToJ5GiXTWc0gIO4IidzF8ALjAOHz3wab5ttV6mrgUuh0zlP9BH3h1..l27e7W8IdrGdxiX3CqMi6q9JuL9M+leMHqAz+nHykjaDGt7KElrLjqS2y3LSkoM8bY1xfo2iwU46iosxoSxNGPQcGUsAqgTFTY8qCDQrWmFPhSeHsXOYvz7KaACd7OYXVYck4IEuYZGsdlHVaA+LYimC+6z8VFSzo5NoumIO0m3Qeq20PWWc6PNPCzEw8wtrOqerPdp+TdSdVsyMmtsQqG+PzOiX5GPPnfAjT6yFjim9SbrayxWz+dzm6mA2+R0aPVE85wDdzi4UISX83Gl9xKWFFfEI8Xj7vOf.jO845I3M1kNrv7BWU0drywxtHAEcAk7iO+ykaWQeYbtQ8a5ebNvaSvjssXjp+hDe6ErAlPxuphJTaq62TPym5TX4Ke43Zt1qEctyssSKvwb4iBMr4s9vkZ53+8acqa8nsIju.AtnbQ.d22+7+SmycL66bBWQaeE+W2RVLd9m6WB1.AMfOMRWZwzYGLRC9RF0m2WtqzVA.Iy2eRl1Ywuj9ELkkxzjErQ4Rujk9xZLSeG+xoltY.01FJJJ46QdIfzx8mFvSbb5BOJl4WswnLYLvAnn4+XgQbV8meFyblb.JOVsU7I6xNtWRo.YCEF+RN0yRBPpd1i7pm06KhmPHwXnF7ulmC0Lo54I9DCy7uNQsT4O1ftcUoy7OT8encfuG8yH9Z+pwZjU+sVlMdsBoqDEfNKep73Z9mCHREwC8Fjs+CYxTW4dg0S7wtqR3Li+YzsJ.5eTVx7awt.u+P+B+b9zWZCR9i8ZfjsnarLlH8+.jug.lEiWFfzIxGeJV3ho.mDIiE.k8mjGU+cHI4YRVHyH0NhCtJR1WQD+X5+B.w1dF1VGQeZQAFSiQHd2Ar28sW7c+69NnolZJmVOLTYkUhG+Q9BtIdUybisIDu.BtnqB.2v7m+Dt9oNsWXd26bxDMYKAaZyaFemu82BwkJwClYikNUV3ofjcsDrf13f5AYmwN.Y.O8OGjS2O9ZpxH5DiAtHGazzTU.HYjxF3HmE5WX0P5ZOf1wAZiVj8AZKGIYASAEHxOnLaTF8IGG56aro.E8HYjedwX.o6Ycfy05v2i9Z8h1mibeu9Is9i4a+r.U5U0yyRn1YF79c5uYGPhZj+8LzO.+6.fKB1QTIWSu8vnqIiMYsa6BeC84wOV8uIXEceuV.BETfV+PrKy2NEdT1oRVvgouv.1p.I7OXdSFKnkQ4+bF90fiRAvAUP5Om7S.O746K7NGKAwiNYbN0.NluEkuPCGGHLuPdgiG+PNdQLxLk.Z7oKPi8E6MrwE448zmbSR7UZaHebvb7IYneLygpDf79mbMCMU5eqN1gCbfCf8rm8fqZZSCsEnyctyXHCdPUczSVZdqdEex2qMg7E.vEcA.bm25c1vB9hOdEcpp11w76AO3AvW+u8uAG6XGiulrXXbJC0N9d5UNNOuzYrxKFsYHNoT65U6ejBe1Whh95E+hg9LuosJINwHCLY.5vBgh51QtLQ5dKV8Q+HS.MgjensInzIJ917NZJBJ9ms2.mG9FRCzpvOvA5itczAXndVwwhP2rAR5fzIYubPHv0MCWzYOanK8S5TXPiu3fFflm2TAvkbJrQ3wyALKfH4Y3msTaFeK8k1yjLOq1C8NQV4OWvo9YftB8j.WV5qeVU+OMN1jQs98MUPOzuHN3EAP6vyx.sC7yDfsfK0lBeGK5SUaRNaiLJsX4aI.MU.YBBP8diCdXGy2yV8E4tNO92EnEhT8MkJEmbJFxIz6fpzLFfOEVUuapWuD92irUA.rss1.hhhvXF6XyztkC5UO6I5Rm6z.icco45W8xe61DxmmCWTE.v7erErxE7EerAzu9zm1DdM0TS3a70+ZXm6bm7.FiCcdPjS8RgMqxV5S5KAIy6E4VVvmBFfoixwJGLPpeGNqKEuo+dzK4BmMfD5Esj40O8SPJxhuC1sinjDke1VNy6qRoCkK5.j1kaHXzslUELIi5rigEeVGDR9Iu2Fm4dAEo+IGfkvm.dK.MlobVUqD0gsrBN60E8WFw2DLhyEwzxd7mlbcp+gp3ynB..f.PRDEDUiCXyEk7wXJMXtnHkNIN49wT+NbxOcfam1K9H89I55RfpP.8L5oGI47v.o+TLTyxuW0N70ekUmy5H48TsCdxYHQKiSMp+2e7u1ylpZC9UfB7Xsru+zVvWeX7Dh97nxLNkStiY7aZalrihjguxgxSB15OZOgWW.k+8ee5SxH+SIx.iyXw9jvSkRoeEpSLP+CDJlZFcH8rBuH1InsinDf+ZqudL3AOXLvANn.xa9vvG1vvt2ydt0S0mdsvMupUs61DxmGCWzrF.t2G3w9eb2y4NlvvGZaaeeB.7S9w+Hr4Msoj+fCJlxFWdgguNmnMMmosNm+DTREEcozVMJ8EjPemA31N17C3XCAB8ErCWMhRpWz0GfHT1BkXbUeBR8dWTnuPSICZESlIXjXMCpdQ1lAdLTI.HQ7.s1wPSs9W8.I5GBeslMD+6KcA5+BYizGzOi1+PfGUNpm07Oo+JAWpycd9ZiiSOj2RdlnnHtPNQQNyI.KcMWpA9P126Hvm3qPm+5D+SUTf36X08Y42Y8s6q+70q.A9aOdzlAe98+lcJhhGcZ56cOBeW5ELUPxHSJm4oCt8e+wfeF5Kd6UMc1w+pwxDonyuC3jyL.862NW55PJMC7rf3jWMrSr44QeqZJ1S+kJ69u+m92QQQ.QNi8IIg.KuQqwIQlUy4u9YbdAylB+K+y+SXaaaaAj2xCOv8euXB8eX00lQ77X3hhJ.Lqae9S8Ftgo+Sm6ccGsYbeye2uA+pW6UkAZrMC5WnKQQd638dO+BdNN+0tfihh344W+wcgbPJYN.9uE7sY1Z93v3T36mcgFnU4eTRVlz13SySo4Tlfo2KcbfPNecijIsfi7xIY6Sm4ttBHvbcg0y37M8AbnsiOQeWF5mG+yBmnJ0YcpUwJeLZz84egNTEfhMHQNaiUFDIi4wPVGFI+WjJ4vzr4ULCMtftRxz853mWmE2oG9x37jwkZ9MM33XJXEUkCPV4mqj.yI9i+rcIF8er52g20.XZwYmp5a3wzjiZu9OS2tSMVgtpZrnyid7eQN34gqp2SJC8iyzbxyxhe50466ouPLTmsHDNhFN1YoeNqeY4cmbzSvi9TGBG3.+9WX7o1lcTmNXJ32M.Nhp.8k5mwDLlvmkJUBqdUqDy75tNTUUs9yGlpppJLjgLnJZ7jwOzpV9R+6Z0HddLbQQ..208LmM+Ee7GshpZieJHW+5VG99e++Qya4RIjTN8zFF4eWasQ.pD3Z6QQQoeHe.3uVWZ7owp1WPbP6zz3X1Ce1IDSUKOoSsilWWeiH9efeHiUVGaNks6.u3wNoT5B+WRUxJo3yVBde4WSKOZRUuH.9nEouJ3DojCFi2Y.emKwvz+X6+hfCdUGx4.Om6.++Scu4gcWEGmI9ac9zNRfX4CKLHP.RHPHKVEqRH.EP.RH1WEX4M.C33.I1wdxLNSxL4Yd98Gy77KyjwSRbV7VbbRbrS7dFuafvRhWvrHL6FiwFCFvrJPBc64O5tp5s5y498cunEDsM56dO2S00RWcUUW8VYzyhtbLfeEK60ULXBqAf7+4gtMRi29mcB2NvPN47wY.eXgWPuTpD7XxjcMF+6zuNvtrnN0h+Sod.hDt43X4WvQL6eUE77Kzxmg2GI+6dCXKvIEdNPy13mz+pzkUzw5u9JXWQpheZd7IXr9RT+etBBzj3rDmwR0bUs9idtAj+VttrqeXookS3ZYIm8AK3FsMqQL7B0dZUaBuu86gjOcTk5ME5aIUzOKHRrnLH+ielrOUp2W5EeI7yerGCG4hWbep+tKybm1IL0oL4Q6gIOx8dO242dfAb6zxa3C.3btvK+G91W6Z1ycaHm2+e8u9Wi+3+++efW4UdE2mIEotYjIXLfcjzgiV52Tc7lllbJ1Ji7S.rQcGbVTojxADXq3d6vBx+M6ys7LUJVNdE+vFBnbRg4Y.vMBTw+locJxcx.m89D5EBFtt3Qd.sis5HibhZVLjXvOtLWMPVpGHcCOY7gcj5+q3Fu3JfEiLOWy+fa+XvHKdzhsqogVvkpwZEdwVSl.vCHfcOnipV0UTGzM1601OnM6AvGUNfdJwgMa3YAj6r1cOkfulSz15DZy+Ij8BjLAcT9kk0przk+lQ8nOtX6WMiQNFXcPFbRoiverOpw+jLxjcgrRzO34u2dQ.pkn9uWAgAnzOXf3hRKniX8oKTvd8R15B.oX8QRFRLR8EqeuZZg5+n1AEAF869fKZOzTAnAbz1QcE+GH3D869ayT5S7DOA5Af4O+CrlQGyxdO6Yim3Iehk8hSaJexe98e+OyPA71Yk2PuF.N6y9RdOm9ocpG5vNu+aZSuJ9H+Y+evy8bOG.HaEj0N1YkEAOo.2UgURGwlSMd9QKFCKUDe3bYvSF2U3Ept3QeDyNVkyeZN834msGn4KLE6vLhZPtZD9gqVS92RznZBzeJd5eoNNTigcR+vG8UcFFBx6jYrn19Ty3Au8nta+3JM3Pnx3pm8fHELRSSwgUTlys+FcBfdoZ3KNBQhzebClZMnrS87xKhu8MQSWxuZ7ukAdcmo1N6ONewBPoO7utlDzOyOOBe92pk+AmLniB0Mpyetn+xiv0o2jAZmSO.hKvtAC9X+OoK3k3I4oGXN.utJL5Wker8i.858Mq6+KizX1lBqoip0Gfo+mLPcZV6iSmW3sxVpj+mt5+mz.tFww+XzakI+pATPvSxu3oNXBe4uzWD2wcb6CBFBky9rNSbHydt24PC31Yk2vlAfksrksaG5hOpu04bVqRF1694O6+3mA+vevOvGQnTEoH8bK0xTT2i6h9qQ2FcjiewGEU9cc7RAR2B+J7cNpg.cWQOhu5dCaEnhCRaNks5zI.GGLt3eSGctRWbRGSTmPmxjJ7KD9CBfDCGwSbahVmiG7RE7b9BCdTQUjXDJ0rZXOyuXjEyYutElDnozVeW0DcCK2TMHp8OgHew7JORbQ7ytI8XaV2MGFg2H4EWUCocn3uwyOyVB3UCubPItNdj+Y4PM+KL9Ii3F7ZeJa5BzLpkythmwAWuID3VcaLpddGNL6JHxbFiz1ekpoNujB33BO0uNf+BQ4Ymp76EmWQ8DdzyNqjB0Y+oe01fl1coB+UwXDDbV+eQEjDtRHLfJqE1HNzM7or9nTzqBGYvUxz.0HB+RwmwsofGLWtMXc28ciEu3iBScZSqy5tqxjmzjvtu6iNwWbCa5nu269N+TCLfamUdia..q3L+YuiK+Rm5zl5TGJ3t8a+Gh+w+g+tnCkvVPicZ.6cLk5pfE72UxJ7ZGlJmy7kdiVmlwByIiD+t5cx9LpTt6XT+TT4ZG3FZdk8pjwuaPvqUtCMILT24Evh1npoeEBY7wevYXfqLGBsbtmpgG8E9vCqcFvhVRrprc1Psl9Y5k04uumGPDc7JnMC1BnijFAG51pZt3vUu6VTxtQJFEU30umR1eqeN2BpWLLtrrL0OaAfGMZ+GOPh17eltcdpO7unt+hF80if5Z30soHRk6+NQuTpSdyp2EJzVGTPb+Dg9CvpGIBfFwRxmfF68p0sX3McPkfzsol9HlPHByn+3+5jqGkpI8BcBjH+1B+47NYY1A946funTaOh8.8aUaWirW0G7NVRM+SvyxuToOFG3PtNYFxr.Qx+5N7L5i1v2vF1.dnG9AwwdrGWm6lk9U1sccWvqtwWcdSbjc3Nuu66tumAFvsiJugbJ.V8EtlOwYupSemm4L2ogBte0u5Ivm3i8QsuGx.f3cjyOqnnTrTMVWXE1HQ0sSSie55kDc60oNIhvYiFtn3yNczNNEJvcD50PjPzgF1HEEYwTnUizMPn6SbwL54SyAMxDwkQo9Q+5eKAanNgE0fjBOymkZLfev32oIN8khV4l3v8hXF9H3MiBvPYkQj3myzOMhToA1H9Ki5jyFpJa0sPmOW8vGgaf+k.7A8uF2gdtILmNTorn7Lm0c.u9bdglZ5N03Woc.HRS4uCO7A5uQBAjXztQ+vVrfIhAh8+b02b1PX4WxgmZOMi051MrrPBMC7k1O0mf2WBgX4Zkk.6Gb76K6w9P+EfC8aM4WA9f9ea3sCzGpieHnbgHMBdqeIE3pX+kreMVzuHAaD.41pv1yqXagn.E8TeYu9b90Om.PGzOO.K.jCtr.e93ImN4B0lrf8XxlhEbCaaxwmKMi3+m7v+D749reVLrkUbJKG6+AN2OC.FtSdtsSJugKC.K8bNmC+jN1i6CexK6DFJ3d0W8Uw+y+W+w3o+UOEf5bFpAeokCddD1sxLPUIlVtFqdQ4YMhee1achIE9Z7ZNvBcnZhvW67uTZHZQKYixMzJ8uK7ycd7JP4hFhdrH08JfLnFWnZ5O6xnJ9yjecsZ4cbEc1WfILxoD43m+K5Hk.zyYT0h9KQLnKfOKErti7dV.LFEX+qqSfxHkYboN1PIk6Iy4n.ej3tCkLf911D146dC7cYBH1sW4yMProkRqubFER1MHmPhnwFdXGeqN9Qg1n.DT9rvWin8mDd5Fb9OQ12U+w93vEZcDD2JgZFBz1AaEkKRIy.41OWGCk9THLxyvn1aoqvNNn1nv.08eK052iWa0Z2rjNRUt+WjwseOlWDuBz1Tdk0quCg9n8KgfgweEe6K4yhMRekoVKfn.bbFHHR40TRE8A3CPwzaH4W.+EFpd5A3ZlCtiuWBpKgKUH.7vO7Cg8Z1yFyZV6Q6WtOkllFL68duj0ugMcw20cb6+ICLfamTdCW..q3jOsG5sdoWzDFl8uI.vm4y72i631yK1CatyHEi5NIsmesNzfn.CDnQqlBippNPB25S4iMVulfiPeP.tiLCdlVnz9q63.EXA5Hy35h5czEO2JMY76pX2MloivxLpvc3BrZzIdM9ESl6YAHPyZkAhlCzOSqcYjJ9ycgedEmGO46TQpekMqUQC4X0cfIVZ+EoLx1RaknsYEdi2Y.0NR0rAzijOpbwoYuUxNBmgWG0AGouGLZNNmx8F.3CALX0oGHBJeVqSMwK1tHPZy+JN0.ADR90Bd3zePWoNP.SALQ8USHraB7teQmD0elD7d+XuMAhZqHJqh0IQ2k50fG9fRTYioeRAJGqSJvBEljWWbuGlmb6.jNcQXjJAYFBXgbTZseU8uTdH4LPK923zZaNhCelr710Tn8oDDtZyir+FnE2fjwqJYka+8eysOm+869NuSr3i5nwzFh0CvLl9zwHSXh6ZCl7S8i+w20+1.C31Ak2PME.q97VyW3rW8plxTGx48+tty6.e6u42L7LM5wTBlRKuWxAHEdDUxBNMJk7d8OUVPMdGAM0+ErFfy5vx9jE1PfRGH.envKEajJ2xfjS+jaX2cECs+OxcN7H1iAJj+csORzPRpzeMYoNm6boOyFghVABZgeU.n7rMZGwmVfnriFkdhpLySHheukgct8SMNTNs5niVWcjkMUxe1QeuDahLSp1Bcq.ucvQUzIZ.JiL1ugz3irU0Qp1tIU5OpCbM3NEW1BmRJYOfnCiaEX3WL7WnULVvCasAXYZnfybPCE8OxnpVm56qetK92n+Fekfmpj+bVn0IryaCK3u3TvVLh5hxz5n6ZcB8Yx2YacnvPYgoywNWrGaQs35zBh0g1+tE9SZ+ewpe0lTqNfIGFlVbzmnWyTXqveU8VveSwtkg2h9Hmwo1E2VEO0d14D.T5jd2fcAFdWnHk.RLBTW3pczzvzRqaoSVPipGSU1K+xuL9n+0+kXS850pVGqxRN9iEy6.Of+WKXAKX5CEfuNWdCS..m8ptfkbBK83Ny4e.ycnf64dtmEehO9Gyi5q3zxmmJezp4hX+Qm2r5Bm5I+HzrXXWzQy.XFDUihryQyguhRuSAJv38RqFMNWJ3WcZnq4fDimhSEK3BpdUGtRM+a3uPOZGapBr+GQVFkV3e8y9bBxUqmUDgf2nhjWeJcVZAXQlaQq.i8WM1BkUISt4m0juaFfO5ceanoNTxusNmz59ZWmF.e0sWPKMxIoLu00yqdp3vqA9JvlcpmaOg4D1guZt1UV2dG2Aq57keuFnWA0LdjPp6Y5kg2lS+J5WaasUQNoGIoDkMAuslqWk+40lPCCegmaDwVOD8L4uFrbbQE5si5NxPriTYcsBn5T5BGjz5ccKUGh+LGom02DVMv5qttmp66xOcD90YEHZuQcBKVcXdGo5yoER.X3Wn2sB+pAQoxdDPPOQ0crfCTmisVS.jbPIyfi8LN0UguaSgAOVeJN05VOuB31qNgCFA3VTUYn85Ju6VFDA3AefG.e0u7WBCSYjlFbdm6pkCcwG65FJ.ect7Flo.3jVwY7.W5kbAiLgg3z9Kk5g+xOxGA+ze1iF6jpFSsQU5Ou.IjvUmZrHhj2mpkAUXmzehuhV0UtbewCU2dpIo4si535ixfnGZGG3mzfdXBMBojWE5u2YDA7J0uaA+Q5OQNt0ShqJ5253KFNIzacF654svuP1bAH7Ua3ntQpE5c3g3G0tI+j3qoowVA9djYBzkdlFNnTDvBha6MdD4pyZdt5E.+TXqXXxa2RA3SEcD0Pruh8g4PDvGUsVO9716Tu9dA7EfmzKSISWVguK5Wpvu1lqzpx+nDvBn1WQcnUQ2MBQ+BL4GpgWDnaGQMq.Z6PNiL9EYr19oyLlMEA5TuTVbmZe95cPf5HqS8rfSWU+OBu5wkiiPB0a01tiGQb4+3rmo+lJ+CYXHfKIP+7M2m5zqnoUjPF3VsZROiNgqaTVqSccIQTOMl.o3fd.ELuFLRhoU19jRO4R9jCrf+DY+gQ.0.wzOfae1p+DQyh+tOv8e+3fNnEfcdW1kV7W+JSaZSCSZRSYm1zHSbi225tqabfA70wxaHx.vpO2K8+6YeVqbhSYxSdnf6a8M+V3tu66ztTLDQhqpe8+Qcnxktc7yc737QpivImBUpCNnNC03I4AhPVKrtFQCOTGFJenMVD3BMRRGldNhJFb8QgYVNz.UHdLFpAS+EZFbHIzHdnpT63iv2c90mtEzA9ciZArnxK1XH53yTKnCOfsp9EO0x1HDMwqXCrwsuWF8iY3jFEtHFOniBmS4Nm16jHD77HiiihmGQtBulI.8up.roHy7QP6OyRhoFvf8WUv33mGZjF7fcQUQ0oFXaM9Uvsi6ZRCxxbgneFs3+VzOI+FS3E+u41OXsmJUvYuQaGsoVv1YGpdmOVbS+QqnZSDA8vtgW4+f9andcYmAkp6kS0ja6P6CpU.iUyLhhKDvi4NW6+lnck.86JAHBJo822hmr9qcSCpYUIHZJ7HELso2axQKzEqcEFsTIqIamiv3uglLhVN+0BuyJz9xfBDfZenFfDR3u5u7u.qe8quO0a2ki6XWLVv7On+n8a+1ugaKp85TY69..V0ptvEeBK83O08e+12gBtG6w9Y3e9e5yByoiFEXHBv7WpWMogEFRUgubSPuRJ90UvJTil9bew1VyNMUmhzpXMw3zijsS5nmi+d8zq6yRm4Dr49WMbmol13WMbYlin.S79BdGT8mL+rhWA778qi+IN2ej6X0nlvbFsCLLKiVe1nEuvHg7pU53yZ2Z91zS64Gug63a197m48xdzZTlK8TwCKvuvphmbb5ygNJiF1auT3sEPGHmgjC1H9KOy1VAo15OIOyDASiD3pLjEkp7OATL1Fks1ICSE90Wxmye9zfz+MKaBU7OS+5Z.nqfAZCuPqWg3YPfEThjMomcc38+BYOngXxp6KCSHQxOirBBOPOnB9f7qz+WeSpuhFvp9NNeTjMvZhB3x3LhNageTg+V8kbaCIP5ebvLk1eMU7.8+1Drq0JkRiZ1NhpOVDCjMQ+25gjmgUPSEgQ9cYyVbFh9Yg5.D1AXjL4oe5mB+ce5g6L9QjFbNqdkxQcrKa3OdAecnrc+T.bBmzxef0rlKZhCyE8yl1zlvG9+8eBdlm9oIknrhPbz99nJTiZscZoeUcl60m1wUufeziBTajJpiMKX.E+NtUbFNLhTk11VWxotr7N5VBSwus5c0+S4InNC8QfqNVC3RzQRnlJoSPqBLwzuWpKkWI9Scv69PEW9p0kJmIq.L+mT7k.gypRD7R82TbhURqaRu7Yxo9mk+paAkCzEQlHfRuOkRdgmybcjlzhcij+bZ80QtZ6Qd.eA4g3H8cAHEoTP9QixTk+A4Wamfp71trVHQJqoITcjnJvZ25C9CseD8q0TP+ufdl+SbFU.rson9WHhudZH3AzodK4iPs7e7VKTc3ZYv.d1B3fArQHFbhmhxUi.b4HIvLAepBdSVoxMEbxYLmQPsY2ZGjn7ya.oI8f5iUS+A7GAG155QHYrEoQxdV7H100Y50wTATjt16p3202DCm5hlloeK.gREX1uTcGU2WnAbI96DnCxtTTim5+nuGTazI7XO1ig8ZulMl0dL3aMvoMsoAYjQlYpYJO188iuqev.C3qCksqy.v4cAq4itxUcZScpSYJCEbeku7WDOxO8Q5TgHup08FatCq9F0E0.G+K8.rChD0.ervcZLqC7f.nNCtiV5sbZqBu4gtRNmK9IZnZP67FClHEFk.OpqH9K+N8+r.noJvMF0dwRZcv4nApDsbT2rrH76rQVD+rANMpubjPIfTOHkz6pywuICC7uaDSGMnVz0QAOBddsAKjNiEfIAOuP45ZMEyGg0dp0Ufof2HUyP6CaLkcZqOSERbaKoGy5rRGHwbn.otoqS72wKTL7KvxbPe3eQ706fVBK.QC9TX0naqQhJ7KksKll55n+It+e96cd+HnGvPkLBTKCBlL5PO0pegaYbXh5OnUgCcO+Zt0gP8WUWVaHI+Upm+2PVGjNvgP8gM4tyG5T8jsE1ACPbBa2woeWXzh9giag6WnvZAak6zZSsaeoAN+m0s+s0e7.R.9a+TeR77O+yOF7W6xIrjiCyad62eN1NeP1a2RbG8xNq4rzi+n9X+Fm7INTv8SejGAerO1es2Orz2yG4o3aKKKZP2PWWNc8eGkikRO455H67nti0q+LBORD+9HmEBjJ5n3MhW.WdjpdvL7dFul9cGJ5u4Nn4wV4NNJuVHCFQGmZn8jeFhdbqh1uyAjYFnXCMb8pLiWUA1xfCPmae8EsE4W4+78kuZTUedAByau6z2kGIGdynT4WJuWiDIYc2ADvuwyjhxXJ+Brpg+51OITWb6eTvEFcdG3uh.L34.LBzeE9iMMz2zTrVq+Q3W+rkIBJnSdQEx2pfp7mO2BxueFGFrk1WKXBpt8r.ni5WIkx6XQqqm2E5hC12gKfXCqAHD+dJPyN+SxbReqFVNXtPeCsOK0OvjeTvAgAiP1LX7GdN29Qdq4QraroP6ZCAcmEdp9q0ULakBgS37SNVlXlHIgSlDEe2jXivpp+iJqh8ev.84W4UdE7TO0Sgi3HOxtYtNJMMMXO2i8PdwMzakq6Nu8Ox.C313x1sA.bxmxI9fq8xtzoN4gXg+swMtQ7m7+7OFuPIZMNsi4uCnJOrQLustxnqZroRwfO8z.n4Zkbv.gvu9ljRtPcVCFtMOdTQGZpEoexRiLAN.hFHCJ7sbfx7jOhce2KXlWp3eZz8UOjgwjahD40Ru5PcR0iEHP+LjT5j21XU9g5A3i1V3sIJ7Q4u6bNCutHiTG5J7ZfUpqIM0zpgsvg0i4TwaGhxp3bNxxGGdzM7fE.oPchJGjpS8VxZdDiD9M4SWvmZiqV3m34.+K7taodDmZUEo+.+Kwch.2OH6DRrynffykxu0PLnoqTh7SCBPG4ueKLJf0erQNJBzrKwq3+PB.Zo61t8K3flkejiRSOJ3XjfgBNPYedcW3AGGwOANzfC3oPHZ+J1TC3qDep1rWPmxMNf5VxBaz6NciPVYcNveVxDiJ90CIHAk0CPxa+psawE8LRvCtkntZaEEaZO9u3mic+M8lvdtm6Um0YWkcbGmA13F2vadDYja49tue7CNv.tMrrc4T.rxy5h+cN0Sd46xNtiyXnf6K8E+B3we7eN8DWoo3UJ79gQxT6zE.5kjgpCam86D7rA6DaEHADSmcaExfw19wTzRVt9Zkkc1nHM1eJQVboN2bG7fQlj0gzSUl24Klgh1NKhFIoioTECtmM3XUM3qNVHzh3eMbnzLsp90CxGdgcAyrFucEIi0M5pHWchn68YOUz9gYSl94UoOK+7zWq0dJXnNXHlcvYxeVla1lI30.p.IyIGdjLOvij3LLxyNveUKdP7GBjne3ulmKetF+r9io+VQ+L7p722tkN7bF..RVpg4ETo1lpseV+mF+XEV20.5YLfFdntfPySkTc+OJSfDI2pyr0AOpTWOpXOjTJ6GZEG5+3xSs+CGOAWWBpVeAUxek.z2yne5K5yT38EdrNEKtLumFbUuTGaQPR+S0iEU5IlDmke9fW7.JsqrXU9QsM5ZQPkeAu6VcVI+BZ+dvAp7SI5+9O8eKdtm6YaUeiUY4mzIh8Xu2uuvPAz1vx1iY.XjUeNm6Mr5y7LFpq42G4m9Swm7i+wByudsim5+pkthTje2QZ7EYhse+A4mhqWydc0HgyuAU2to45eG.dtok743uxKVGOn3AVcoJzI.593NNhxvnOLrRvGdWh9SjLsk7ifoVVa7Wp0uUK+BFHbhi7A5Fd8t3I5phswbHENflH7pmLeb29Fh+rVEp8aDVtkn4Kth9qG8pS+A2D1eq4eF9lNgGHm8fvp8fLn6GpMwLb4sk0ABUq+1tOhLlzeD+co+v7xX296NT6B9r9i1+C.clU.0gY7r2mnaM6KIy0lwAJ7ITMRZw47rdFcFBTCuQ+vK8wLFu2SL86p9ndaYxa+T4eUCfEPM0WI19YHR+D80ToeQcerZ3o1uRCXCUu5EybilI..Zd1J3gBXhY.dAZ1YfOEbJV6OsqHfq+ZGWv87EVMK0sV9VwmHlzMtXGyvuwMtwgdp.FYjQvNsSyXBaXixd9iW2c7EGX.2FU1tK.fy4heq2vZtjKXe14YNyAFlM0qG9S+v+I3Y+0+Z.vNJbkINcR0qX8V8PEZDN.kC6G+bfWCx09R.E1...H.jDQAQ0RPakRVWAZzJcVphMUO0z.jrRrNGW.f1wAU3gcdAwcbnQ+aVIDqSuE+K43goUvvK7uEGQZXjsI5cq40PUpOyMzSDf+2Z7qW+q5EzSU6n15pFwUCDV7.MR6asNHL6yHKKyAhAQTQ+lNjIZ5n8OXXmjeJ7iq9S4uMQCyRgNClnCzeT.pAwzB+F8m5D9Vzem3mbp2I+W5+UBrMhem96u7qM8WieN.gvzz.m9s6qfZ9uKZEZyZIzZw2cA4eWrKJJ0wh2WG8uT0uwdUeX+F9c9l6So8aMoVA7Tnd0JMzlPS2fAe3c41Z9Yk9u1BT04WKqLUv2dseVQ+prfds9ZqTkIkLEpC3gjdnWud15+IqN2cVHhCLjwJQiTihRpO9u3WLzWXPuoceT73+xe4Q7fSbj+jm6Aevg6fEXqbY6po.Xomy4b3G+wbzKYel8rGJ39le8uFdze5inAv0citveyiTLp5UJoD5939MQmM0UcHJJI49oIsZJ5QhQatxu6Hr85KFgzmYKvEkJLGFtAIhyxOOE6PEcvo+SDdsyTM83FHJxQweOcTUFzRHXepBXZDQwtP+st4vLPQppkU2umteVlJQzAXGVLt7ySYe9uI6497IWPUJYKxIq9L5O45apAIwyMg0VWKBXVlzW4LPfJ3a4MgcVS7erAve2XSfzF+rifwC99feLVzerCXG7uGHj22qevqifrM90103tIHJ9zifX87aHUNOGjhdRuJ6Gg.HgNfV5bDHe7BlWfvk9E097Z0+yF5JQ+pLHAVIBsGffVgIqYPeTa76+nalPa.noHfvSK5usBDIKKi3NoCRIeH8T2FDDfDdbEFOfjVF.hfY7o9Z8f29kuPz5wa0GzNK.pjIQ5e1HkZ8V08f+ze5O0Pe.AsxSeEXIitme+gBnsAksqx.vIuzkeuW1ZtnIOL2zeO4S7D3u3i7mYy8SXzfUM7bDspA6xuDemBb5J9WOdRgv2dZtw95Teyi.uE9oQc3iaqk0EyXhPczs51BpPwoVG5P685ODEOCOSaUvKnK3QzwAKOIGeV73JMPNEB7YmNzzJV+SV9aVCKN9sAIINtxFT70djZL2GkPphVb3yifQhm87TaLfuUyh7L2l65BB+tdje1uEzS32k15enE7zuUge8yrdovzBKZUZlZOpai5K7iC9c5Wpnep8ue3Wn1kw.9.8S3GA7q5gT.o057b6e4uwK6FwzeZfV0t9fTCuH1Mwoya962geDuhL9jf2Z5yOW04CfG5+SABTw+5OVGHQT+UOKRfAMT4cnx8FJW7Wz+R9oKXP9KRmGYvtsB8I0GdRU1uTVQg0zET5WkatsGafPcry.BsQlcVg9UGGFuI.u75eY7RuzKgEsnCoEO0uxzl1TQBXloIOw699W251t49BX6lL.b5m8k9e3TNkkOioMj2ze+MexOAd0W8UyewsNU9bx+pvirUnF59W50KkSoDxyCrtGu0Q.Z0l12MhRnNtLkwVQ11wH+Y7WfOaHpwVjYZmZO8ZrIdtZUGm.1Y2cGvK0vaPqufXclBzOwqsbvC3qiHqijJjHq8Udnpo.Oc+MzHkJ0Yo10EvEcJIm+8TxNkFsT+BTs2xQ3nkMveFCJ1ohmwOrXlbJUQ8dLYM0i.yMryNu5K7c33x0l4fZ3JHNJHsRUmWA7O.vyzeW3ODfBQ+PDed7Q23us1Wefmnet2rGUHQTp9AxxeeTod.e5EdCejMCn5DIa5hP4rjP0szoDPWzf5oIndIDg.+J4otQIXVvoeW6GmnWSetHnoh2BlSTYQG8e7.+ZH.f4vjMb0R8G4bA3Ah6r1XCuynlbcLNm.hN8qvu0+so06a6BCB+1BQD9gSTl+UCdwhIgL6nU+JkcOWbK3ltwa.Ov8e+8km5prrSXIX1yZeFtiVvsxksax.vYt5U+cW8pFtE92+1scK3a9M9F12aOmQR32r4LqO3vihUrSbulNVLHgQYPCIn8Hp0dpz6lX7GoilFAoRdEsnZKAdvqC.zEN0HYS7yY7SczFD30fjBQ767RE6ELpGFEC7m0gEJvhhniLcMWzXumYDnVlpvI90.qNxC8BlQoOAtidE97y7zB25D4ykPFeYxT92IG4h86TrNJdIllWfbY6Xd.WsfGQ3s5Tg0LxIQ52GZTKgd8gwyXAeM8WieyHYQWgkUA35.+J8OHviZ7qA4RABXxi9z9oAdLRAmZPgJ9scVPo+mfjcVBXvixBDF94IfcYCYhROCVg4bNnrS+UGEZM+SA338iqZKfX7kHNGyNtb38AFPUVUfDJ7oHsDvOO0bkcRf36tBARHCpsGyCWWvoKE+JMYNnqF4u4XVr5Oa2TecGmcY2mCYM+pov6xC7wC5IW9IO7ChkrzkQ6NjwtzzzfoOicXjmeSoQu+69t9JCDPakKaWjAfUedq4Kb5m5ozLx.JHA.V+5WO9G+LeF+AjxoMZeydQ46AkuthFrzfamy9hM+RLLAEgRT5twGuClaaJ+gNOeJnROfhQlx6KfNa+MJDT+Dm.faiHSKIyHiaDXLfWpfGD707uff3y5.RFe8Q1k7NgrQux+4Yg.lrEkE5mdOtyaqurSC.NeN7Q1a9zeKWu777Cfv8Ye882P37qWDaKfB8uvMNDbjlf4fT0eh7OY3gSwehNlSIbHzu2BdJ3ptvu9LVbyq5csYpK76F+GO3Ga76A5RzuS8lg8tvuw+iA7L8qUf+NZJnc0+XmNRH.W+fuoE0s6os0OYQaAbNSQwaevx0+r3YDvusAKzGrOF6OPxWmApxLBI.nt2DORGr0E.8iiXBbIJyLbmTW1BI1rk+X0r.jpfu37UJ6DfDJamV+NIwlFfVYCfjKh6TWpn+Vk5l0BMKTaotPMLyUTfrN18cVkETAs9dpCBUC57W7Kdb7s9leCLLkEsvCFukC3ftF.LogBvsRkW2C.3nO5i9Ms3EeXm49seyYnf6K74+myGOiZmS0zBGkK06wGsm6pgKpgWKUUMkNzHdWlyJElxFqfP0upDqQh5Ykf7dx3qG0YARXAGZzf3cDsdifnAg3O13kx+8CdhCX302yxFfTsngTYZc6.Pq.i3NppSeW9TNiErsuXiYDMA+L5WMDoOyuQ3JhQ.aTZpic9lLS4eyHtDuvcTxI37svi1nh4Q4qssl3ijmIecGns+lXf0S05kf2cR1e36B+7HBCzuXshch+WKv2M9MAgG7f8tJel5O9Q+fu+7uvHl5OvAjY8+Dsespu3q4CdwBxWxVt9Cn0JBLm8YcwH+aAsp21fRSQGONJaUbXJdISv6AFqxei7SjbmHZ3UnIlA21UjeZ+R8+rpviD2rSvvKiG7ZCXw9XxW.y4oAnwtD0hEwjKt8C56FuKV8Slb7ei3eszzzfFD0uZc4uo+ZHNZwBnQkA1q4FPwW9K8Ewy7LOCFlxJOsSQNmKbM2vPAzVoxq6SAvwu7S6Nu7K8h24gYt+ezG8mh+lO4mv9tNB+P5ZJSLr2zSJ2jYFC9TbaqzT5rp+VhpKeQu3vy+MUFIfYXpZfirYNjf6721hWEGTBrrAna8LsdBKNFg67jHZTwEQ.oAGdW9khcNM3qqWeb4dm0H6pemDKNd00oPYz+42QMEnetHGaxOW.rQuoibScPnF20iCVMEuMA9ul9cR1a+hibzbl.TIWTlxDxrz2cpBzA7IRnDcbFGkWWvSNqSsO5dsmMl3eyAdudTZML0DoJIPGxuAE9NwOarlj+J8GL2mhse9NAg1Y.ZFAJ5OpCiD.cMDW5Wl3ypKedw4fV6oQsVpE2UL0rRCVoRPGk+L8SOm4eUVpALXyUu9uBK+7i8HG9zvCOwGpsJ8B6weVYGSzHjrxK41+jaVARfm5p8yzKRo.94o1hu0.Mc6.7kZgZ+fFgQ40XZkg+Ue0WEOyy7LC0YCvLlwLvK7hqeu1vKOk+9G4Qtme0.C3VgxqqY.3jO2y8jV1wcr66tty67.CSJ0Ce5O0mBoTuvy0tTZiiMxSxIE2wqtno9uo59stWwXrNBbSwOQ3RXm6pAUUyAlCufFEUZfDN3Z.DpgIU9thmRcn4.TiP0YDCGVWVxXpCOFW3AAuDvOUurSYy3FnQxT8Yox9FzQ86ihJGHP4P6oTq5nrrHx0EmE7QuYxRPGNIvamZ.cXiTLj3NJXouwPtgUw4eyojP7tvUPHryPcFa+b4ZfNpZCC50sfOhOilrlOxgK82tv+.Ce+vuvxNh9GS4mNBywGdywizA9Ccej.8a1FfieNHQHhouv2q..91ayyrjuUC4yRBU2snw4KFPTxVfZWIbmUTQ2FMW8W0dlYayo+t3eUVZc6LgKCuuh8i5O5yFG30EGm8bG9FiH7.CzcVQdD4df.0EE2p8DcpbUa5l8Ga9HztwtwkZmZVlGZh8KCFscQcrcwben8A7FIEu+vev2Gq6tu6V00XUNkkehXu1uc66NT.sUn75ZF.V1Ir7ezkdQm+jmvPbU+da25sfu024aYJB7eccB92bkF9e0h9a5BvKqv1X+lPuiqWmqSd9o7QDKdGFDowZbmQb6Qeqmy+RM77eDILnw3HZQa3ofZwqE36D+znRCvKQVkYepMJi2FjR4CuCdJBMCavOHwbiQHr3HSvWfj58zPB9AATWA8wxDPuSqQTR3jYkV7eWrKYfNJ.TbvvGy9v3Ae.+5HlAZ2lHUbuEPZen+AAdj5F+7HR6j9IYTW5OI08I8NzeY5uF+ZaNKk5DdqJn1eR9EVyHcz96uSFdc2BXWwyfR+u0qN+WKS.E4ZXz2cYafYFJZ5H8Ww+1yn9ucDcg.2doBiMH2xyDpFZAu1+wfgWrdZ0HgLmpvamlpcdEBWsd.XdsB+sx9fkpfX6t00fnu7q2F+F3p7S.o+j.KjXv+oOxOAK8DF7ED3jmzjPudoc3UmxD+AOvce222.AzVgxqaY.XkWvE+NNwkbb63TFhq52W9UdE7O+O84ByoC..DXovm+O+Jvr8dfstWcudoRT5MVjc1o6URMl4i3GZsItxnGIrzAZp5fSQ+pK1P.cjpYZsG7N1pOUKnl.6qFAz2S4UFdeTDF7JeHCJ7032eVx5EJwQ7CqZLbwty.f47WGgeVdxgDgx41d42DeT+5nxxKVSZj9Ht39Tit1HH01Nk9s1OeT8b.GgQf3Oz3C2ujX3psyTzBdWRmFS3QE7BieKPG.oba0wzjRiRQN11.KFJ36D+TlBpGktuVFb4aK72G3UcaqcgOFmYYSJJ+LcG0wD09KDCUq+ZGGyhePAomADr9Efu3fEQ7f.nrYkoAOHfFwuOO7fby2mE7JsW6FE5..Cfh7KQMN5hVMp+FvNq+Xc7cwmmoDscBjFvX.u1kmwe4c8E.ntVJPouKUGsxB.WoQcLkwbcy76YC9qC7qpg.hsNDFqhy+h8ceAkWkE.ps5W7K9E369c+1iYcWWV5weLXO14Y82MT.sEt75VF.V9xOsa87O2yZjAMhI.fu7W5Kf69ttK3iz1Udhet7WqCQsRVaXxzQwTr3ifTLk933yr.Mrm5cJ7QN6ueqRJiyZigcS+L1I7PzOX9moJi+6Xd7QejeA3oNA76I0x7nQElrTmoZcqzAek81Pc3z2uQfcx936YewtdXCiOIP+jye1YjYD0M72M+SN+BvifisP6ecaAOEAnV6oO3uCm9joxNg2nUo8aEnevlyGa9e7fue3WBPSzes7qCb1UPGnz+SkKcQ+RgvbGSwrZD5+GqbC+skevBn.E3yqM.X2.cJ8YAZVTZCRQIH9LBvBnWC7.dlDMRzXRRnFn+Rek.+q.pp6t7Kd1941OE.a.NCG7cg+T3UDq+lK+x8M0fcz07C5nvSQQw5Ug+s9RB+tQ52wu29qqCgvZ2ptcWqEMaTE5uC1OaKipqexC+PXIKcoXRSZvVf+iLxHXxSZhS7UjI9qu+0cm21.AzV3xqKY.3LOmK9OZ4mzILogI0+O0S8T3a70+5EGOMwNFH9YOh+bCYqiZ29znqUherVlUV7U8IEgI7tA7y0ewc5Ug6xbQoTptZzs8Wr6Qof+jspaUmiQa0E7CxPmZIpzwwc86AzjX3o5nM7dFTz2nK7adxcwnK+LoDrSWQWbHTJ9cYtEPPxOLVzQOvvyqC.VY1mq+jOBxhAkVNq6xQqvLP6Uit8ajyFSVPvaICQ0MjNvOIeh+KAOgiZ7y+lAUE+aArLF3enguZd36JPAl+CADTg+wBdu8qa7qJkAi0D8GFcsgeOfDg+qQ.N74QxlejcFQnupRyF9yUf9b8vCRoMs+mMcWk0Efs9.XzyBiD+8JidB6rUYKuuTbcPHN+aQh30y3Cu9algJB+RU6c9KZl.zcYA6r11MF8IS.piey1Efu.qMQgzwmccDcQbyqCAFKDvQ92T+8sBXj98OKBvK8RqGegO+mGCS4HOhCC66r26+6CEPaAKutjAfUs5y9qelqbERqntGixm5S9Ivi9X+LyIU+ZDBiLzbJWYvgmyTfvBvKmpORgiavUEcJhS1vI6DvwaEOJhsh+6wOKw3pzoJfeMpc6SvBAQh6gUyPC0ouBU88YMCM7wQVGrKn+Gw9oRuplxIqlFrlxAZVW.D6fXQmKet5CWzKhGcu2V4idwkOZMPixNE4oH7R.9.60kSLfPJ1szwxVvq0+Ro.8Yva5tcfKNKA03O3nk3Cl94Q7Q3eXfuE94Tn2A8yxuvHto0OvfBOi+V8EZ7V4tneGbG+F8CJH4w.deD+99tWr9hdPAMjbjNZ50F.H.gcQfJ+s5x9mtJJ+mXsqfyYMnfwh+0f6iY3abfOz8wmGdHTf2Z37j8KqOHslIzodsGu.fnARzVF3o62xdfoTnzkP7BaeLW50qmgGSVzw5Av3eTK+h9OxYxI+4G8Q+o3vN7CGyXF6Xq5qqRSSCl1zlZy52.1068dtyu5.AzVvx17L.bdWvZ9nK+jWVS3DjZbJOzC8f3G7C994iiUwU9HcU.nsgsaHsRkwSeE+6UDevUncCsUhZkWNyYKbCatuk1zQ8YLfHhsBiMcPUSxGhtOB9pNhdJrTSf9vs8rOnN7nNDk+wkehwwoZ3k9CuISqPUvhj1QUONeIXhYSoH+oUlTNSLYQRbjWL74uqoh0bNzmTFqVjEFdoV9MFvWgeqtJ+BukwbGU9Hn6G+65OZ6ejVpg24mJ7WI96j9spIh+gA9Z7yxLilI5mg2UcpjI8A9wB+bapJ+poesd5B+A52F+5XCOH3yYlB1wIrcRRl7cJfc5TZA85575IlW1jPpzV2zA9YBORCFOnzsATw4tFidkASQsoQ11FH3kZ3ES7Ua9RJsIYdrfC5u1TuFOGuYoLq.5Up38e0VUq0Sx8kU61dFbxuYSitd.pcs2tv17UcstFDpJy50aS3y8Y+GGmZMVVzBOXb.Gvbu1gBnsPks4A.reyadqc9Gv7FJX9m9reV6telzP00slUBQuJ0yaE8dk+lOq+82QcHmcNCSKVmeO+8pLVWpQ6HnMfEXJ28.BG1O7sNWO.aAV4zOg+LGGluNG7T.+Y8dxXIQ+dDusC50cnxJ9sg25TVUB8QoNGiTcN96FhU5t77FoLBt1QXGoy7esy18x2GgoeKMebPFvbPj+XjIZFG3qweKSUjShVGwtUuW.+LLT6O5PVFfuhfBNz6C9Sn+3enguE+SGiUCB82B+UvWJMiC9U+U0xO62aFG7WfWpk+76mPq9i7cKgjRE8OQeI6jCbDcyvWAOeBEjOzqZrS+R9TQk6+MVdr3bc1E8qLZ2sez5pI0e9mG86XZ+H.ueOBX6fBn1+H4RX5Qi0qx59MPJCJo+V9MK6K.183AKo.jfs+ZZtuOWGSDRsreHkyNl67ttSbe2681Yc0uxIsrk1bdWvZ9nCEPaAJaSC.3Luf07oW9ItzwKnqP4NtiaGOvCbe11twMPE+bqnhQzIF.P6T+2Xq5+xK.tCB2QgG8gMm7jQNNBzV3k1yq9g8iO5A.Ze+SJXQ7S7VIxGitKBfDPH.APcb45LdeZqxtNn+9.eIBCRfSRuPvGYd0NVTghyR2XmAxq0uDrE3m9t5EfT8g8itm9aLzyzOHckTkdimfUNCNPFa3qcrHfkeTiQf9c4RvvbXD8Q7auCASmvS32nA.qtQed1Xg+gB9J9uC0gH8WK+qvemvKcCem3m8wU2+kB5Kq+63mgGz6v0itm4qkep9m1Wt9XEVqGMS.RBkr.jhmoEleOuuhdJBVieSff5uS1J39ulMMN3e8sx1uLssTzAeM7nFdggmaq71AOng7izc7jOhbw1MO5.j564C.zfm7ShUi5Ia3beb0lTWA10TGjUmknidS+grekECprJyietO2vkEf4e.yC627l2ZGJf1BT1lF.vBl27tn4N28efe+M0qG9m9reVzRKC..dpw0LA31qhQVBztANOuS8r2OYUfXYezwDL7SulgWa9yrXvq6gxB57uomy+YEeEdn8CoWsqP+crHJgn3Wb5Q+iaf26Xy0fEPg8zJYWGvadkHulh8SkGVnqFV.VpQ+zSSS6eA1RGL0weVVEOre7WNaHtNflDy7rfs9qIcaBEcnzE7sbPQ32k+fL50k7q5YV0zE9Ga3aw+0zU06J846A7OLv2E+KN+2lWGD9evg2wal+UmaY+ncq+lnGHspGmOaw9FwQvX3Qr9u02xjfdNevT0Tzy6kX92OrqrAlR8qay+ATP1mx1ECrf6szszjETN+Jtir1Ka5Z7KE6TD7TvDlYK19CUrflRpcP2wYT.DKg9UVfjvnWUOo8Hy0eyqAov+i0sTnCe6r.Z7uEPfKC.R3Q9IOL99euu23V2bY4m3Rky7BVymdn.ZyrrMK.fy97trO2xW9IN9Rapbq2xMiG+we77WRZCg+69bwj+VJwMVQTomS.ZoG7H.sT+mzyXa2YpUW17M3K9DOB9JGuJtoTZkubgP3hDgiJFhFHiwUF9cmxcMRkD.ieaTMfne3NIcIhS.k.Bb384VyvmGsS6B2wWSiaYkMqx3dI83Q0AyREZiOBJ8DOrqSdQ9L624+JC+ZvMkmwxOKqMcAO4PSeuH7k2uC3A+NvgybTJptU0Hk6K9gA6.CugeDo8N9sszvqzZ3cpneajkCB7nBdq8qC7KNcxY1Ppnk7T+Tgeuhr2Q6KEfG8A9xK1.+VmTO9oSLcA+rDfuXpzolHY8279H9oGHkE.U1jePvV.a5wncUeOUr+QVWb6WtCRKP.Ty+L9QK3MwRBT+tJ6OzyZ09a0bxCBnioCvo+LOI5eM6WNNcaX42OUQ.ZPK1BwlxPIpvZ98orNXpsB+JlMFke97e9+IroMsILnk4N28GKX9y+hv1vEm+1p..F4fV37O68aemy.CvF23FwW5Kl2REZGoDRcsXMg2hj+SWa6uVY.H3MKXhybflR7nS3sNhSOVTzNjdorGf3NxN9qlCaODVOx1jGwN2QJPfAvkfC5jJKDo05X.z6pN+Y5uNZa98q+aE4Wy9Hg70lpfjux+CUDrrgXi+VJc5JO2lGW86QnI5rl+A09y4pI96AmuzuEfmVyG0ZO1eCFk81GOsjH3voF+0zuAeSS.9.+qFxqkA7Ha0mqoCm4mWiv2UuGlv4eugvGi+TWvyUCktc6gL+qwk2G4WLvkH9CsepCBqvaoMB9D6thPPJEzeAptoAEeQBhRPBRJYKPvTRsj3mhf8R9pj2WS.IygahYf.Qozexn+Fh+aCuOZ5V7ehgO7yp3Ha+nxtCJvxSWPvlBnLe5M.fD.cjIfDBM.b6IYGtlNx3Pbvs1yboooouYcfgO6OvYe19I+Ws7KeheItoa5FGy5str7S5Djy4bWy+vPAzlQYaR..m04e4e5SZoC2b+eS23Mfe8y7qA.6DtcTZgO2girtJMct.uJih1lerZEE9X+0+NOOlQjHEbIT5rkvbC1f3BTgo+7nMbkaNheGaBh2G2k2UrZvBJHwxOshElWJetZ5.f1wIZarB9hQ9l703a9FPSGIStd5U3671cloeO89ZV..DOcpht59cA.6LM19SzuD4e9cYZtF95oSHCes9GIJnQUz4n3AZQqAiOFwTkECC+tivZ3EDMLyAoXzUsid1YNgm5fbFK34.FrQ6W9c0QRM8Gz+p4+wBdw5ADkyJ9oqgOS9AV90gtRM8q3L.O5K8yJ.0aaSa8o.mVER+qI.tTVff45W6qna6UMK.Zfz5ZBv5+S8+BdWY9i+oRP.wQtKjOUsuhO5YyuIEfT25ut8CWdPuaxsov1+B29oRInfFgdNWTpjfg2EWJwpzSPVyzs2qzdVYGgEB3uF6EkBKqPU0e1TgWuMPv+xW4qfMtwM1AuzcY+124f4uf4eNCL.alksIA.L+4sem+vbc+tgMrA7U+W9JpLF1nsqFlgzwm6x4OaD1WQ9I69pViDU6j6177s.XxB2WeNrmyWOjVoLA28Jq5eMUS5h9ypeJpTKHihhjY7pzwklEBn8ZS1ykpQGkf2gGkf.nHtqiXWnEGkJwXi+FdczGguIupZKN+0T9q3WP6fryxeekAamveEzpxMezeAA.YvJYNOLGZlCSl+UvI4JOLGE9BA3vKcBumN0TnNUwqmVx36wzm4bWEwUABzW3Y9nF+F8wxApsibPnsebcNVvC54nhtpke0zOK+p4+wDdE3BxCbCkEAm9AMkeoV3OP+0lUpful9Y7KD90e22IBd+dQfcfVQRvRVtxPVu31zgBXqINFVRtXlk3lHJPf1VlDuIydUu1T7qABG0yopHYh+3ft7J.p8M01Vvng9tE6WprPu906gwY94IAPhXFW+AjMsL.r8Oucw2lf854SyQ8TFyB.+2c4GGbaNaAYZ5YdlmF2zMNb27uKcIGmbVWvk+WOT.8ZrrUO.fUcAW5e9IbBG+XzR1tbCe2uCd9m8YCJ15ns4n+P0m6ba+IBzaAJc+epcN4+yUZ7N5pCYC2EOY77goNr5rTRemdJTIhun+zfH8fIUEHsdY7Cyvr0qC9Hb75fxRQ9qw5sz4zWzWBPPlhvm0NQ1mQ.8E3i2LCpQ2FxHlg+Rz5ZZQ0Q.7QpiwK...B.IQTPTEuM+7lMT5Xx7uZJVEYRQXwFvM8A05iHA9Wc3Zx4VvCy34XAu4LffOLhXPyaOQ+IBOZa.iW1H5XAeWzOO53TE8zl9IZkLfMVv2B+07OKGI5uF+07eeg23cX8E71EDjOL8yseg9OUOW6u0O34LIX3OQ.xxAl9KTpttUzoAHIFGPYCPL8eaOxib3i4rATtkRkxtoI0qPyk9dTW1ZBgkul9RJY300+ppfh8PWux00BYITbaFIuFM9WwCa+JXkVT6hhgGU1XWcvsBDPwO0cwYXS955eN90T4a4RwvcF9FJiR0KlPk6RIX7Rn0mhIValEQvW8e4qhMrgMfAsL+CXtXt6+bV6.CvlQYq9hMXkqZ0e9y3zN05Vv9Vd4W4Uvewe9eF1vF1noj4tQxEyYr88pHUqdWSeP6TCE9js3bf0nRvUTNc0.2Hgl58TEbZIej2pJIIC+Ho62buhMvEVQpnX41Kb9j3emu4s+hR+HP+7H6U92Lxa3jdl4E.A5PaWrn5k797Oe2FjJoqLCjtndSvctqFCE32de5n+6AP3WHYBMhLPx7TTtnCIhcb0e9uRuQMtoxXqEuqQIVapWwO0Xw3uOze.tJ72l9Q+gmpG1wEpooTj+C3s5cY3SiE9q9qk0fJ5uK7OdvOd32zaI5mUY45DjsDC+CB7ji.12eP9Apmden+DP49DHp+qGS.w0EfElq0mRD3YYqQPJ0yn0NEjNC3Otv.g95LiA5rFQhOiTzc3M1W0uSTmkL+aOqfQ29q2tzqLngdE9EVvEn090GTXFdlTqxTG0H0B+Rr8OeihlPSIKlAcrJa6N7J4G2hjdlhES9+xu7KiouC6.1u8ev2AbSYJSQd4WMsi265tyu1.CzqgxV0L.bNm2k7eaoG+wMT3369c9V34dgmm64FifqQLENuDGEK.Yf.vmOdH1YRs10oo.qqvS+moK6QO5JNpRZDubg2C694XOa3QwE6pgvOTmApBGOJP20kX+iKITvM5unPxihl9ERlJHHdIK3lMME9xAegMxjB.5B3U.rE4jtR+EoLheKK.978mai7E4m4LTaOU4evQJbmUH7PRb5sui6nnIC2riKF+01Wi3W7umZaLodTvllaG3GcgmZ3E+2.UOF8T4Ltl9Y7x3mgul+ize+qCzAdL8Oh+6K7bvXcfeRsuy5HUieAsoeAQdtK3o1DdTrsn4htl6cvTzb8ZQO3ffkEL.MKgkmU5qaMAE82bl.Z7cJmcvIw5ALSyxl15ud6rn+eWlnzu8L0NI+aBgGl+IaDojAmI+gFHfiuvMDnYiV2sVLyvEsOsWuARRsaozs.s2lEmqlcVURxmNfoTxeQ30EfO0BpcYljruWZ+affu1+2+ErgM7J8gOZWV3BNPrO68d+dGX.dMV1plAfS4zV4WcUq7zFo0JHsOkMrgMf+xOxeN13FxKZBcDS1MsEIba2f2FGJ7iHbZaxupNOSriQ2QGgiDxotNEwkFAbW3UmxAUURo+T.9vfGb7SJov3auiQj23ElHupl8NFN+PHqimEBfRslPN9YVM2L3K3O2XRFfD7Ewj04jlSW0npcI+.e99MCtDe6zlSyJ1b6+dmtHeyxOlE4QMTk8DLfv2I9cGigQg2A70y8cW3uevWq.4F8h0o6zpBbTMXw9w+RPyna5mjiCJ82O36RlFneJCEcR+zb62I94HGIi6l163BuGTR.+kJOR+w+B.SGONu9Y7623fYpws+kqa00skgfjFDdFq1fOCBZ8YN0Zznoqjh7e48hlHz91dUkH3z2kyVqPB.irHaroVBvjamL4sIB5Jc7pLgHYaDOs8QTCNaawGEe6SF1P8v307WH92M.q4+blsmwzmA1u8avxBfHBlvDFo4kSMuz8st65lGHfdMT1pkAfy5rtn21QeTG4DGYHtte+WuoaDuvK7BVzh7BgSi7lGENfJqIuUk2k6f1qb46j+Mmo04kWfXJ5bl.3zJ4Q.Kj011kFaEr52Q89d.V5PQIyr1zUCG+wNeh9pfzqrmEuOt49Uscl1V5oK1QkwcTpcdLwh3m8d.9J8OeXrTFgu3mmGIwC9hmyeF+55hPOc+L6yJ8Cm.3LXXFmUm4pdhq.EbTn+tDvdzX9PCupqQ3myRA6Ef+M6YZS8XP+fp2fSYF9xemzjlDt3K9hwwebGevPuoWTAqJS6B+AmYcv+gfWnLETOJaPvUCuLFvGn+p1G868S9qGotQ7K1p1m0cx3Rguwz+BAnTqeXxFwZ+5j9Eez1P3SOvLrZ1AquwAyGTP5hCjzQk3YoQ9Ga.O0FhQrfLKZcfa0+Ox+5n76f9qru35F7HhE6cSTEn1cpCTOHqU6GPoA8XSO4GSvAl2oI0YLqeYm.gImuTdKH9DeQYqmNrnwk2csV.p+Nutp7rL39OZffu1W+qgW8UeULnkC8PVD1iQG8ObfA30PYqV..67aZW+iOhC+PG32eSa5UwW+q8uXe2FknXcwJO2OZFEIpFZvlRlAAeO36Mh5Jx2zaxZnbLtV8a1xJdEiQYWqHnyWU9+3EYntaCXcIdTihUUI6OlxJ0ow+S0pati2gS8UWwOqc.CQzVwRZGTkqxLPt1Zn8lrhed55jFeaO46AZV9RAiEZKIRwrPTy+QdtooAyd1yldoD4nvy7Pj+cMqvH3IBLZPoM7FsJjfKA2oKwU8Cd6ypSSF+vsiOdvuO6y9fO3G7+.VxwuDbtm64hcezQi0amiBRLZMf+TL.Ki9Sof7mkeA+MUzeqgfg1xuVlU4fIFK9OEGA1grnEg+f+f+PrK6xtTg+Tj9U.BvSVATZpF+r8GyGZkdBI+reo.ma+o3bmvucvz.Tl1LMXfLp39WV+uTBBcrAC.rnEsnHSRMNs6+3YRjkwduGhyRdSIKyj5FOn69GUFkXIqVAlXIJ+SlCYcWSYGSvcre8CGewr8EK.AP0Iidl9i3GkcD.r0xUDiL79gbU20u92m8YdFbK2xfOX9QFYDbzG0QM0Uspy9TGXfFxxVk..V0ptvEezG4h2oINwINvvbq25shmw12+vUHSvlucHpSs9YR0KVGpdoxQ9q2Sf2iorRt2QsF+UumFVIUraVPZtvyvmCFw2sAvhlmF1o9AnHMDbRhW3IBBFNUJkA23+RvRI+mE1iNHb2UPtI2ughm3gZjueHpCNWO6ySo3ofVcFbTi.094YmSg46Up4efQ28QwG3C7Aw66889wttq6ZvftF1lMBZqtc7ahNJHD14f4LnC3c4u6npN3LqYlpGG8D+SHV+tAO4LpFdQDb5m1ogq+5+swtu66N..l7jmLt70tVyodXjyUxunLfnKKvgjy+cz9w7Oq8WS+032n+J4GGGpEzIELPs7Wq2IM4Ii0rlKCW0U8twniNJV6ZeakXwRAX6G7H77Nvek7mc0D7.ZNzjh9SJHm4K2K8jCjWGL5YI.PN3fTujs1m3EGHz5RJYEPZvLm4Lw66889w66248gi8XOVWXlLTFc7o7O0ALEDE9foz2SZn.pUkFMPmjKuJPaUnpoGLeaNNy+UGnfHhMU.1MzZm6H.2Vcz9AS+b6lGPfG3hTXAu+uYiqWxxtCiQ6ST.ddPTb++jMEChjWK.apGGl2XWN5EeDXm1s2zmXfAXHKaUVC.G5QbT+qWxEe9ybxSdRCz6uod8ve0e4eNdoW7krHorHGsFlboNpJtwvdt1QqrvyZ3SwOB95HgUm6ti1J65Am+DdaDj54KpMfxh5wf2gQ88FRDG8r1FJiFWzwY0xHFbYk2Qi5bWgr9N5URrZ7OYAQDZuIWdc8YR15l010PvKD7oTcl7DutHGppQIK8mF867+RVxRv0dMWK1sca2vDlvDv9rOyA27sbyg5RTGYBMh0Bpq4+5C1k1Nb5C70Nof2VzJfG1fBs9CxuaiymRE82A761ttq3pu5qAKdwGUqSaxcdlyD.I7.OvCDF0ak1a3Ywaju7uVy+A8OJM4L+WS+sv+X.OK+aI+XXH5de228Cu2266EG3Adf1qsq65thM7Ja.O7C+PwQ82A7AaAT6dmser7SqJJE1A4G4fvwuWOpyd0kvHL7.9NFH46DfRK.zcGfNcaK9HOR767676TxDFvBNnEfa4VuEr90udtwKvGF+qiwpl9g2+ieYdvRnh+4JPbTYD.6f1WGA4VflfCUtdp1ICsJ7nxK89M5W+4bMjHBHUUoV.AHmUktOO.bb5zBw+FYK1e026kdoWBuo2zti8bulcepyXYhSbh3EewWb5OyKNxG+wdj68WOP.MDks3A.rrksrc6DW9uw+sCcQKbfg41+g+.bi2vMfXiXTf5NRg88nFs+ClwYwS+tzpwQU1E6u7uUnDGlfifJ7pFM5w6uVTVzeswezAm5npvypSpD6vql+Exwpn56VmwXcUu0G02g3ot7NXFaJyOXYqGA8ULGyBy91trPT65hu9BzyKcimYmDDpCNDnmysEyX5SGuqq3JvJN0UfILgIXME6xtrKXSu5lv8+.OP65JX7Nh+PyI2tBzM7UN66GM2E9qCPnVz2RupOvebG2wgq3JtRra61tg9U1+8e+w5tm6AO2y9rUzOg+ZZlk4cI+QzwXMexxL84sdeh+Cv2wm6p82fuoAqZUqBW9kc4XG1gcnE+Ou4MO7iti6.uvy+7cJ+Mm80.5dD6j+U8mnth6YJDvQvtFr9e0xeEmIFmEv4sNXBZv04uO4oLE71eaqEm+4eAXRSxGz0jlzjv9rO6Mtoa7lhJVNQCqEPb7Cv6.F2Ics8OwrCAn1IZI..OPqH5aYOh.uGnr9oAW.09g2tEvCi9PVS0WuEAnsjEb5555mQw4cr8mjeT+xv1E0zeKelv+S9DOANgkchXPK6xttK3ttqa+jtm65G8mMv.Mfks3SAvL2885e3XN5EOTv7M95ks5n5TrdDtj8.+2Z0kMzgrgtUopuOn0HZiKti39Csw0rqTzZG9oMO1MzkSKgSG+d5nPpNR1TnCNeGlyifxSaEZ0gKjNJilU4UxeOkwBy2b7uLNGoowdW8H8sF776oyYGrg3KoxnZn2soqNLJMK0xuXfaHkvhdKKB+m+O+GfCcQGB5pblm4Yh8du2am0HmRNdpa+QnybLq.w6xwt5vquWP+oF+UeOwN7fmN2V3uB9oO8oiq3JtRbIW7kfIO4I2oLPKMMMXsu00Bd53r9OUzOi+vQTLg+wl+k1F26f9GS3G.9W+9niNJd+uu2ON8S6zak8CsLgILA7Nd6uCKHwVzOQCwQ+58Yp0+b0o15OiW6m5JKa+AUvmuZxkTJdDgKv5+naq1Bof4Mu4g+q+W9uhkrjk1I+ePGzBvYbFmQ6f7MpIx+pmXllC2i.vcplhVfhiduexOejE8o+mWW9zAPsskKUstVO.g9ugooPBzoaOH.dX6.Jhu0MGoiocHvODeZzeYePK5mo2+wdrGC2y5V2XTmwxtsK6BNjEsvCAaEFv9V7J7rO2K3idhKcvO4+dnG7AwW9K8EAf6msqTuk+nG0WWokQbsmB79VjID8WAd0IuZPzvOzFOoM7cF3guebcZsFdlFqFQoZPlgSXABBcr73zc7YxGJKBAZVhzFpvYT137kF+OeBYoqYXc2Qpxacd+aJzqtcl781rFXhy+hUmNw01Hb9cm3jlDtjK4RvEbAW3X53qooAGv7N.7u9u9uZW4yAY8XgeJkebKd+bLEf2+g15uLOUsHxDp8IPqgpL+sEtvEhq4puFKMuCRYG1gc.yXFy.20ceWg.6ZkAiZ7SANg9I+Dh+kJ36f9EIth+QE7bVHFK7uzkrTbkW4UlWyGiSYFyXFXxSdJXc2y55P92A+ON3OUdIuea61+v6ZzsqSqcuiv5G801AjE7CKH21XdWMctm64f29a+cfoO8oOl7+7m+7wse62Nd1m8Yq9E2FfQCz.TBbkQ2wU7u0RSUfZ2y0+q1MWUSW.K6TzGjeIXagZQpxBPkS75U9uVoFuHfxHk9.WTnzs5GnWOOM+ZvD8SF5zul0.OqD4Omq0dHgW3EddbzG8wfAsL0oLEr9WdS688rt67yOv.M.ksnY.XUm6Z98Opi7vqkPiY4a70+ZTjfkHHCAlWx+B4hsCe+wRuTb0zx5wz20T6zB+gRcuiXooQLmLIpdCKyCVQyvimZMkPD54LpzTsEn.sCYA7jghh7q71lwpnDvqbxhKaD1XjHSXeVSvRijW+C.wU7eaYUW7e9y7EvT.RJMZyYNyAen+SeHbBmvxvfTl0rlENuy675hiMCqVGZ0gP46wwE4uembi5PCtCEagCxuasRq4jLBuKWhvOoINQbQWzEgq5JuJLiYLiwg6aWNti63vBWnOsbTrkAmT52GS9m9to+peljmL8WEMPegeLweJgYLiYfq9puZbwW7EGR283UNoS5jv7Of4GpSkr3AGqs+VgW.fAdJ5PIRmHHCZ6jzwia+I+675APKMTs..LqYsG3C8g98wpV0Y12LevkINwIhq5ptJLxDmPc2PmGYZjs8l3WUBulMkEfZ+KeuJ2BAGj52saKDOhJ+6TEHRaGUVV.pJrSWqNJF.bScJcBiNXdTuERyHRrE3cK6HJZJFfEI1elYesqQtMVv5t66F+7e9i0Qs0cYtyc+wr16YeYCL.CXYKZF.NwSZ4ekyd0qbhChRI.vS9jOA969z+s.nD0DE0HO5BsnyCUzcAr2GfV3ekeOQFz0Tjmn5BvmGmZ7qMZQmmdQuo+zsvlRtphP2zuRuNMyNjTZzqCxUoPoXikEh6RUkapwLoB+lEpVNjzG6XPONiEIda9ozOJWwunbNk2Soqhb2V0+vWYukVESlyzbhfWo8llFrxUtJ71e6u8g1w2bly9hG9geX7j+pekMxddzoA4OYPV+MNktcAOORu.7RjGLmhriDU+lwE+NDr689rO3ZtlqEGzAcPCE+WWl+7mOts+saCaXCaHv+1mI7aiNl0MaACyKUKTSl2E+rf.iC7pboK7eHGxgfq4ZtVrW60dMz7tHBNvC7.wsbK2b91YqVW.UNHT7q+VpMe5AAztMl0A.B9gB7r+YDgWwaoOudNAbxmzIiq8Zu1AJyGbYm1ocBSYxSF20ccmUBFh90AjXNrbtLUw.QcAwb7kej3+Os6SotY61grxVy+jcTT9oTG3uqEDnXzuRKfxBPk8vDS+DLlczxAT1XbcAa8+833bQZAGNFcMsWciaDGxgN3aU9zl1TyFFQt66ecCw7GLNksXY.XUq5BW7geXG1zFYjAOlhu027alG8rjihRBBKORPARbNUqbDmMvjAz1un5OkxQbMh30s2tjQPV2K0tds.fiJN..5M8W4L4DLga27eD8Kjg.iJRD9C7EepCT5Tnm3dH4zuOzMqFbbHVDvT2XB2DaIFZb3E+ZHMYzexNW+yx17mzQsvWpOpbPSW2HECj1ntEmAr995WJkcazQwu6G3CfUu5UigQuhKu025aE6vzll6fIQ6u8xb.GjdjdVhnOg2y0CH77uyAhYN4StrNr.5n26zVwogq+5tda68s4Tl9zmNtzK8Riz+XgecNxYmUf4+XVDpfNLx5lp16tfG00QA+SZxSFWxkdo3JthqbbS28XUl4LmItjK4RM4e8Zvnl9CihmW.rFLD+nQhyfSOOz9qXRWSJHtxz0hex.BLycZmv0e8+13xtrKanx7AWVwJVANnErf.IXDZpcKXRY.K.0HnoPiVGmkIHYhEecDPUh09q0sqSXjj9FhXiJWyHR66H.uXAsPv2B+1.qb5OA8XhOYv2.oclVZ0Vm7lUR9vaA4H0Aba+a2Jdtm645KOTWN7C6PvLm9t7gGX.FfxVrL.rni7H+1W7Edd653snjzxK+xuL93ez+Jzq2lrHlXgpORb6ITiX78TCrMksimt+yjNgG9uYOSJQL5NFiyOUam+ZPFthgX1QaX3ERI2PuPvTsEEsnfKQKxzjWAJ3TPC7nnixOG+sLwUhx0qKMR8lFcjXtLnQ7mUKSUGY7M8mJiMZkZqByCH865yW5RWBtlq4Zwn61nsn4goLkoLEL5n6N9A+fuen8W+q5Xu1oHO+vAi9xPBuTsht0eq.TXw1AXs+iN5n3ptxqrys22lSY228cGO+y+b3QezG0nOk1Y7aOqJvk3e8ENn99r9mxydbpcn+Rv6yYtWe669tu38bsuGbfymRe+lQ4M+ley3IexmLm90N3O0Oj2dS1Zp4i.OAW9Ufsxjl8Et+SSK30u3ve3G9giq+5t9WSY9fKhHXAKXA3Fuwavui5clBYzSs.JAX5z76512X9yrCHVsDBDRoCllZ4HtF+zua0OPYqV2jGzQHVfRMDZ.7EOdz9KOkOhMfMGGYqhJsZYxp0zbIVfLllu3R.sY0oCfTudXZSaZXdy6.vfTlvDl.dtm+42gW4kdgO7i7HOxKMP.MNksXVVNjEsvCXG2wAOEs21sdK3ke4WwFjgI7EWAwSqh2H5Zrkmj7SFvdkqe27cKeFdaAi.V41a+x32y9fg+RCYWoXxW0+plWxt7JTEee8EPcC7fDsHFSkHMsnfERkoTAd+EyUNUWI9Ktg3jGQKieqTzQsA4UpKc690qjYCKC.vSrhckkJzE1QQlqN+c4txGU8Qg2glaUm9LlAt1q8Zvke4u0wcEtOnkC6vNLbLGywDctoijmyJQ4urS6bSR42pVqBiK7Rj2re2fuJqDEbcrGywhe22+uKlyb12sH7ec4rNqyFitaiR5hQmaNey+lObOW94zev3tZfzjCR.OL7fgOQ6DGQvJOiy.W20c8i41b70R4htnKBybm2Yucgo+B8Z7cUlQz9W7yCswcw+5KUmwAyOBcz9nNUSIL4oLE71daucbsWy0tYk4CtrK6xtf0t12FhJ39eCSGpwb0Z.vFn.32Qy1kxpA6OTEXv.yt.6bGvGHh9i9IIZ7nCuGRctV.LDDry2A9q3Qgj+.ZVGx1CGjiuGM1HW9PzgGEf913F9temg5fAZwGwgicZW2yO0.Cv3T1hjAfy3Buj+2q9zN8iZ21sAedo93er+5x49uGsH.YrsTr4hgdG5GMEVcYxxinVuomT30QkyN5GS7KUeujYAc9fDowliNdU55V4TEBAdTwZU6ywDGonR+l4HwhsrTOnUmU.Zz+BiXxfUkbkp9xeyq3ec09yAgo7luVGREwgSaZjxMLd01HizMyC9yId6srv2BtteqqC6y9rOso0Myx7m+Ahu+2+6i0u90SiRQL5ndzI0NDMcMP1NL4eGvw+0Z+oQDQxkb6mfcXG1Ar1091vxW9xCmsAaoK4CLo8A21+1sEoQiVaS+daXene58stSIR9AXx6.7zm02YzQGEWyUeM3HNhirk8fsDkINwIh8d1yF25scaPY1.M5jeq1eM35.8W98v6q8mK+Fy+w.9hePw+bm67vu80+auYutO5prW60dgG+W933w9Y+rhytpWHzN61hL8WaTx9KvxAsN7wkP1vE+Gjp5u7lchew9e7fpxOIoocMvG5Z5J1GKilH98RHTXnCryOyEjX1FpzM804.K.jH9gOPTfblv2q8ZuvdrGuYLHkYLioiG9Qdj88luou6+kABfworEI.fS5DOkO6pV4oMgA8V+69tu6EeiuleMGGMh5NWYG176UWzQrym4+4slm6XzSMCoHVUu0olpxUlGEWie5BJkT5niB1wea3MmzLcAkt7seh8rH6Gne2jIssUHlMrHj342iryiP8lfdy94FniqHdEl715K+Eeg+4xAOPHoh98Jxo97wV6EeQWLtvKbr2deaNkILgIf8du2GbK25sVbboLSba+47Z2zuMJVoV9SiHlf2k+vMxSvqC9XAGzBv69ZtFr2y1O+B1ZVl4LmIR.3Au+62oQs+mRT0zOZy+wLkToqVI+XmBcCufi83OdbEuy2E1kcYW1Zx94SIvMrA7fOzC58eTZsZAaFxxQ0Z5vxPYQOx2QGnS4mUOg1eW92HBNqy5rwa6s811hMp+tJG7BNXby2xMi0+RqO5Dz7v5tb84RmbJapJzJ4uvG1n9IG9lIn5o9K4uiTbZ6YZqC7mf4TFnjI.gl5WRTqAIjH5TY.NqDZlFhso96kRY7j.rrMHizzJvIUauK34cFfv7O.dgW3EvwbrG2X2fQkMsoMIuBlzCb+q6Ntyw+sG6xl8T.rrS4BNzC+vNjoLL25eemu82FsC6DfLuT9VJLZ85hOB3tgWu7crNrV8PM39qO9EcKg.cgh.Cewa3thhmRSFN7nPcUb0VfDglsEquoa2fpWAV3.jhmgeyXNK3pjTk5n0kskMR+ReKQBo32Wo+vlV.819SomZ7TOJ38cN6K9P+m9PXYKaYCTSvlSY+2+8Gq3TWQamx7n7X50LlEoeazd4JvLV1BdNXPQ3lKyH6jlzDwEdAWHtpq5cicZF63VcY.WVwobpXelybB7NH9foeSNwN0zQ2PvRp5VM30eGvW9sYL8oiq7JuJbwW3E8ZdQtMrkybUmI1q8buBl.hxhxyBses4euOD2MqzIqJaAVf4U7ODAyZVyB+d+d+GwJW4J2httO5pLsoMMbUW4UA0OsUTO7Eml19pmZ+E5079EcH6zQOXvQiBW8RZ3O9u4fII7WjfRwliEHfBUmGVOJ7pehpr.nTp1jRPJTcHheHAo2Tf1kETqhyG1hLWHcnv.Gxe5du2eL94+7edm0VWkCYQKD6zzlx+8AFfwnrYmAfC+HOhu5Ec9myrl5Tm5.89O2y8r3u4uwuaCZO5aUAI9ag4bJ+vvn2pWLdVisU2jI6RCRJUi+DAO.pahSphmXi12N1NM3A0gOxarwAUgiGUeqQ+q5mhIP75gLBEkeDbf36JVwwelwZZZrU3udP9DxrPo9D.apUzi32DP4HWFw2OP+Qjl2deqDui29630z9Z+0ZYtyct3ttq6BO+y+b9n1TZr0bzVIaMwYPx5vzZj9saa.8968rmMtlqdye688ZsHhf4t+yE2xsdqHkhgzl+aWzOMBtvb2Ww+IxYWG7OC+BO3Ehq9pulM6E41vVZZZv9O28G27Mey1HzpyHQgXseyXJyEK...f.PRDEDUJRYc1fJ8.5ywfiPnNX4UB.mzIdR3pe2W8Pu891bJiN5n3UdkWA2+8e+dSj3+WHKlFyWBJzdlKA3UcO6fSJNboAxWra3YOH7Nj8+ZKm.I6fQRChPWO.s2pdsoe9xIx11hVVHTRyWLntPIW50qmWWj+m.VUaJndKqy2c.95KqGRXBiLBN3ENXGe9SXBS.O8u9WO8e9i9H+Odxm7I2v.ATeJa1A.b9WvE8mdrGyQ0cvPcT9Ne6uM9w2y8XemSyuMJ8fya.20CUXiykU+udTZFb0RACHBJamGoBWvBFfi1LTJy+eS3T+KWG022.d79rS6Z75NxYaoFch5.DhNUfEQazXZj9c7GjGA72XcFZLdzoIMv5TgnZrXIbi671.zFoT+n+FA69niheyey2KN1i4X2pORm5RSSCl6bmKt4a9lQONctl7KNJGyAl3ifgsUVpf9CO+A82ZZvJN0UfK6xtrsoA+zUwNk.uq61kEk+ioeeDvzcoo8r3o6mIypBP0zuKfOoIOYbAm+Efy5rNqsYi5utLiYLCLkoLErt0c2F8aA3wK.vp1exBi6v.vZ+M4G7rcEjOkmsi6zNg28U8twxW9xeMuUW2bJG3Adf3G9C+g4SIP2OE8A0YtXAEDJE8ee91caPAmnj7yLg4+jsPv8LE.OKAVVH75QgVGLROZ2eoAf5XIR+g9uF94SmQTEfSQGW8eXzAUAUENmX9VDLx+vHWA+xG+WhSd4+FCrNvzlxTwC8HO59+iW2c8ONP.zmxlkF2pNmK35WwodJm1dLq2z.Cym3i+wvK8hu.LgQovNt8GKsU3H.34QRcjjH4aWv6NPIk45OSusUJoyRDZduCJS07hieJoYlxoH9mqgIzAQBDTH.f.uIRmuSbXKnpt8y3eaw+AfToyql1eHR9NKuzwj4eMKe5BDTIeK04Uz+RWxRv0dsuGL5nadauuMmxzm9zwTm5zv5tm0YVeB5h5eqMXiN3Mx.QvoO+NVcmvttq6Ftpq7pvhW7h2lG7S+JydulM9YO1OCO4S9DNCjRtta46gQGCuOiNZY92.fGf.PH6.I.ruyYN3Zt5qAyeKz16ayoLm4LG7POzCge0S8TcDzhDoezgtsIqH9WWneV+eX0s9NG9gcX389d+s1lm4CtzzzfC3.N.bC2v2E81DsZzMugvbvWmpC29qmcV9UD6+k+g3NKnV+Qq0jVk4ZPfIyTaY9EEjZeKO3uD.rqLwX5Zb5GTSohecvmP.2Jycc6g7oCnfRlFrLJfftAH3gAeE8ncEfmEfW8U2Hl0drGXO2yASWXlybmvcbO26A78t0a9OZf.nOkMq..V7wtjO+4cNqdGFznVtu68dw27a70CcvbeU0QF4NSrGZezMPom7epiR8VnKBuOxLPMpVjlP6ryXJF3fhGRk1ts+Bvq3zLNP7GnQrWdmPDmjIjXfKdVA3HbsfHBAZPum9E2OUod7uzzH4tSZ8T9Wd07qc1DAzsRle8GGMFRsej.XFyXF3JdWWAVwJNsspqv8AsLm4LG7v+jGFO0u5WAX7Tz4EaNy9MNvKN.PMPIFdqyeVW8XOliEuq20UrEeassknb.GvAfu2+9+N1vFyYTLn+oAHY8gX8WVwJN5JR82fqQDb5m1oiK8RuzNu89d8nHhf4O+4ia8VtEZ+w6Nmz2g4eKHfpemqSUBvYRRP916aMW5Zv4dtm6qaY9fK63NtiXJSYJceJAB..O3+fsETrXvN5xuNbO8bLUpNhvQQY0Cqxw1Sro+0Lyyx4Lz1zRlRQmsvg2ZhhHx4Cmjy0Y464oXPrW2x3En10f85nrvOgDIdq5yh.7Ru3KMTKFvW5EV+Hu75egO2C8POzSLv.UU1bFBxjVzAuvceXTfuoa5Fyevh3JQYoNYOWUPbSxUMXzH150KU1C5Iau+qMdF7h1MjTHUm35ms.bqvGfep+YEuAsWApTGvq5Z0JGlIgjheUn3NZMdsn3lhUfG8boBxJj9gcgwt7+IbEjG8eudoRZ8KJzF9S1h9SDw1CrRBYYsJ+M9mf2nwLOrnEsH7G7G7GhCcHN1K2VTtr0bYXpScpY6LD8qFpkt0FfKm8Q+paz3tfe5Se53c9NeW3hGfauuWuJSe5SGW7Eeww.IEO3unMUl+iy+OOEBlbnnNN5nihq65tdrhUrhsax9gVxmRfWR9fcxnaoa9mV6HZ6eK8e35+rCq8e+2e76+g98wwe7G+1B1ZfKm5odpXAG7AGbTEcRmpdN09CNAORPuok8KaD6pMJe0+mHXfls.y9Y8PnKq4nTYfXE6e1hziVQyb1IzQraAgn3Wa+ZnAkU3E91PLedDj7LNqTaGYAvCZos8e19s184du2ebNKbCX4vNz2B1gcd2+HCL.cTdMmAfy47ur+zUdFm5QLnaWm0u9WBexO9GKb6r4iRmbZps5pCwNb9qcP0ypdDfOFck11YQ9oiPsNhP68Z67ubq2.e1OUkbwZjAiG88JuS88osT3EOqAwJPU2TZz+LS+vLtn7gE.ByBNIC+1QLCqJ+7ydqBkprLJmA.Id5.fQig46MLpfbYhSZR3RujKAW3EdQaW53SOk.+g29OzEYT6mxapgIW2XL3+J4vBN3CFW069cuMa68s4TFczcGO2y873m8ydzvnVU688m+K+COm491uAHAb7G2wg2w63ctUe68s4T1i8vOk.s4+W7E6mEHfMpspL936+KOyAku0zzfUu5Ui25k+V2tIyGbQDAK3fV.twa5FK2UBsdARG2gwL9I.rS5DI.73HhUZqockFjRHCKz1BjF2sGfV4Y5hPF.HblkajBYLTwOS9bgxDPoQ2r+Z0BGYSseiNvYv9uAhyAoTBSYJSFG3ANXKJ3oMsohG8Qer27McCe6WymI.ulC.Xom3I+2uxy3zl3ft2+usa8VvO51ucji9ibDwBFgZNs8YeU..16YC5pBdPeNYNZ4eWZjpprM7VIkmmLez87necijdvCU3BT5zImIrEBtdU9sgo+XzEUzIEjfpPivO6AAk+CFow2u+InGnO9kxCfeJGx64eH4LszTw+Lupede228EW20e8XgG7fsxVe8pLqYMK7zO8SWNTT3Q+57b8gaTXz+k2wa+xkIMoIgy6bOWbNm84forcXvO8qLu4MO7i9Q+Hr90u97CT9uDEfOJsZ9O2upVFM8oOCr10tVbhm3I85xhbaXKye9yGeuu22Cu7q7J1yX8e.pKVW6ZjfywbYVyZV387d9MwhOxE2sMlsSJScpSEitaih+8+8+81NDK1eb6ff7xC29WsMJgr6UdQV+QeGEFV+IQe1z1nXL08luUaRdPZoxIBaemZdi7iArXOG5mKbamAtTkU3N1Q.Z1QxcepNY.qDe.49OO8S8z3jW9xGX8jMrgMHuRCdMeAA8ZJObye0q9MunEtfoNL68+a61tU.nMr5vXqYxRhfJBeoVMrRH2zzfll7J+muDZpUXT7YAKTFcBmhpNk2k.E50qW71Erzv0Xibtd9qX5OE1lKBh3Wr2WSoH0QypT14OhxuBrtQIDG9QJpPKRikElFQrENqtqV6kx7nd68YsKhDOdks.apbNBfUspUgOvG3ChY8llUGB0s+Jm+4c9XW2077x6x8hLOwFYX8GMCVd590xdu26Md+uu2ON9ieIai4jM+xjlzjvZVykANiGrtXa9GzXXf+N.XgK7sfOvu6G.KXAG71VlXynL0oNUr10tV.37T8n9iNlpBJn5cW1IrL76868ebqxoa4VixQezGMN9iqLOzbmZ04KT62D+KTZ9MiczVrK48O7oKwydJK6baUw.M4fr0Dsnm4H.hcNjXSC.PmmM.F7Z8qlQIC3wAAv9jhvGlFKQP8z.zZZlAfeR01NvBj.9UO8Sg6+9u+Vzc+JK5sbvXpSbG9+afAnp7ZK.fIOiO7aYgKXfe+m5odJ7.ElxG0.nQMSYC.h+UCh1k74Ue4lm2ZQQ76bsFrRYtlKueevhcg+Hg.LRlW6rhQJwW1CcTeB8isvutEprPIrPCS1+3yqdn5rQ0W9gPdypdeF+RikggDbE2jFPPGgNy2zeMF986a..fce2GEefO3GDm0Yc1ugXzdZYJSYJ3xeqWNjlRXdUMfsZ9nuFzdaZvothUfq62551hb6885UYVyZVXOoUlda02PhMC5uPDLoINQbgW3Eh2463ctU8zraqUYt++3t28v1qhh7E8WsxECgDBDRBfP.R3V.ETFu.NCJNxEuiHBBgYPv8nyLa8Ly4LNOyL6YueNO6mmYqNNNJWTQAT4NAB2jahBHfCBF.ALb4KAHbKgnPR9BDR9BPRHq97Gqtp5W0q0axafvkuSqju06Zsptqp5pqp5pqtW69tiC8PND621PJJwuroXH98M9RJgwOtwgu5W8qhYNyY9VhD8aSobBmvW.Sba2VWeBOgBygO2PbxbRVmMQGZSEdtIpUeN+AbmARb3I6RGVvog7Rzj3uRf0zWp0NJh5jA295jpLS9Tdcosmpmzgm28TcMq8TnBRttZ31o3SOxJH3tlyb5Fu6nL1wNVr2yXuldeCPQ4UkV5Oxg+wNuO1gcn8cpb+qu0aAOxC+v1rJTOhBqoHOqKUYR9YZgYvMFqbuoTCZbdEzxyNyvczqOIzddwyw.2KzJxS3tf2bJ1aDu8UGQJZeUwgOSKJGC5.+Sk0EhsC+9wvZUQeyB3XTH196m+1FniT7kkn4dZzA3P+ePevOH9JekuJlxjGdZ3ahayDw5ekWAO9S7DYxVBgjT6abeHY9Ov1NoIgu7W5Ki2+l4udeuQWVzhVDNqy5Lwy9rOCfDk+CiY.HYYe8Y2kcdWve6e6eK1y87M+s22qkxK9RuDl6bmqQjs5+Qb7qY3SD7teWuK728282+l51660RYTiZTXW14cF2wcb69MIGALaZTjN7LHReYUOlaU2lfiUmz.H0RrFQMS2k3FoCddwxevrq3mc+MGk7s+RAFH.u8M720o4Koi5riD3CptPiuzwx.X+1lHWg8LROuRiKavkgC4PNr9dRTqacqSdYHO8Bl2C866K.nxl79w5vO7Cep6yL1yMoE07tuq6LrlL5ZMygRJv73XtBz46TQRBgP9qphTCcpSAZGL040oTEUpoj+aD1qmryZf39IsDdw8dkFDPeFPnYTTdhQorg3dIVweU1M17I+lrLGxCF..R0nFTHnxubkHnNk5HhYoVeg+px+mRwiabiCm3IdhX+e26eIvCqJKaYKCO5itfv3aM4uPl9sIjXyzq42G3Adf3n9rG0aISzw9sTWWia9luYbC23MfTcsaPCvoeal9vdt6zJvG6vObbnG5gMr1An0rl0fq3JuB7a+s+1laPFkBQVjoezHG71FyXvwbzGC9S+S6+sx0aEKqXEq.W+0+K7HPRpr.f8ExKd1.vqwMH8ct7R37vW3YX6KKfVkPaaSGOI+YFkS1yqovz2nmJUrys7RDdUSX6kblO6.jB7214A5APTc2mJfZ8CoP9gqeh1PB3kdwWBy89mKdeu22Wm0WY4ctO6M1xQtE++Bfytu.fJaxN.LlwO4Se+dG8+Z5svEsPrjk7rHXn1lEK6EcvI5NK16V4BZZnXzqUmDH+rx0ou0SZG4g1kpJwL3AjPJQIHGHbFbc01mB0QGwiaTQ6yDHHFfnNelKzVmQJWGRk3KaWcrYMfTQackxnHjru3UH6EssHEDBVlLXuy8cew+sS5KhsZqdi8LreycYNyYN3JthKGqacqKj82h4TUNGHL+7Zt2VNtsDG2wNSru6699lMI7ZpL3fChK5hmEd5EtPn4fh.MZPtiO15eSiYqffIMoIii+3Odry67a82oCanxS9jOINuy+7vxGbvvDNBaQzTp83uTBSeZSGmzIcRuk7LdXSobu268fy9rOGLzpWso+IX2Rm8AxNBjGTjLibvbRj979jgs4E77.i3o4GqmNm.BRhYcjAGtRN04glpW+FAHHet.j8Cssi.riJ7LofebBKJ8w9AKj8krKCinQOgtMo6Z6.ZS0Mqe2pmb6mXZF0PDA2yce28sC.awVrEXu2m8bW5qWtnrI6.vNM029GcG2w96SWH.v8969cPsh0dV9RAy08NKT5X1+0v2Kt583YinI7mtmKaDLb2Y88hI06qk799uppxZGIQNAnFpUijP8hjv+D0tJdPGSgoB72x8AQCSja7wxhVK4Yb5qE5SBsN8m2y+PrY4y97VmPyNiHa7WithPseM.pxseyZ7dr3C+g+vajd+2ZWV0PqBWxEeI3AevGvL5YYxLOPMy+UmzE.Li8devwe7G+a5GkuuVK20ccW3pt5qBqSO.fHE4l8NPNAjgS4KGvAdf3ybDu4cT9t4nr90udbC2vuD+xa3FbG.07aolbBfhHBxQIoppBehOwm.ezC+sdmsAaJkW5kdIbwW7rv+0somUKM6wGdep2bePWj0+wS7fz0CV9oztHAi4SMnIg.c84adw3rwy5rLzIA+ayR9cxvUWW6akat4CsuSit8irid0NA3NxzvdRYmDXGLLc5AFFQnAaasuVgefAdP7RuzKg98ary9r2yPNpi7X9quxq5x1jNW.1jb.XuNhi3suOyXu1jFoee268RyF0M9qFR8v+nEoUcndJAfrw457IXGsT.pzPXLHstSB6bgty28eakJpcxBVB.pE5b.vLDmqWVKIM69TAc5TVJ76T39zVjzpOecwTCQFAPxWLapwehF5rRP13u6mott+.BpDW31cpnoeQOHfpxzxttq6J9q9q9RX629gGY3euJCLvCgKZVyBqdngLi+.vYfrBIwWtjQMpQgOyQ7YvAcPC+xvetLzPCgK6xuLLuAF.IVHR7wG5vJl9UE9ia7iGG6m+Xw9rO8ex.+VwxRW5Rw4e9mGVzhVTbnD4DHx+1dVdf81MkofS7DOogMY3euJKXAK.+3exOFKYIK0lcacRmDP9kJUVEdPxmJTgWiQ4m7aRKkjZD0mzDZ0OzHKpSdJElwLm4A5mjc8yZUyA1SihxZjBNBX1BD1vbg2JIm9sE3PbGAqxvasQ1YColS0OGSaHQJQ.KL8XSIKkv5ekWAOv8OWb.G3GXi2ABf8Ye1aLxsXK+e.fW+b.Xedaa4IOiYz+I2yBepmBKe4C52Piohx7MuubueZMkV06aUPP2imhj66JBsOEQgL3NC2Dp47FfmWCBY9uVRlfbxDxrsHB0tL9yg4WEXzvoqgdOL6eKh.I+d5fj.9q9DPU.QB7jTzjCzcPMkOpr8HATmRM6e678qCqqqlziM31m7S9ovm9S+oGVkg+kk0t10hq5p9Y3Nti6fTI4xZpm5gkFJyTm5NuS3D9K+BCqyve.f4O+4iYO6YigV8PPo+PDoBNxBRnpg2rOui2AN1i4yOrLC+4xcbG2Atxq7Jv5dkWA.j9XZYfZ9oJo.iW7A+PeHbTG4mcXcjOdkW4Uv0bMWC94+7qC000VjNR4kLzmGRhUV4pMAOScX5jzoGwgv2UA4SEj+x7EWZ.qFLPhm3fphwDAdxxkIKukxMptawhQAv0eqS1Q8cQ0+oiEXy4pyBRd8FT30IkVWmBQQxAicIH5Lf6yDOYXA268dO8sC.awXFC1iYrm6Ze8xTYSxAfIMwI+o1koti886eO2yuCl0IyiK0VUgg2dcO1Kb5z3yO1ei5pLvUc4sdOePcuV+esBzOyj9qQYKOsVPpmrk3eyuRVEHNwW3kqUAN5qBipg7R5jSVB1OFqBpPJU2r8XpTNdiiGpivRkfpL9WKM3pE9e3G6ka2jmB9ReouD1sca21.7q25WV3BWHN+K37wxV1x.f6adKAnLezB4eUENrC8vvG8i9QG167y0ccWKlyblCIolGSRxe9Z+mUDmUFN5QMJbDGwmAG3AdfuIQAadJuvpVItjYMKLu4MOebnNFUi5EbC+75TO9sZqveweweIdGaB4A0aEKOyy7L3G+S9w3odxmLeGeIvpQUVGgNYDS8jWxJu7IpnSxAdhVm7WlTgZNRoNI3Sah7xfreJr9wrd1Pt.PJU03mVAf0CexbVX5CKGPCxFz7JoB7mwO3NovKArfbDiawdneK1CBGa6VE6S7PW9jAdnGBuzK9hXKF6XQ+Tlwdr6xG+HO1O9u3pl8unu..aZN.L58bu2yszOHC13k669tW3JPLWFyZUkvLsrwXRGG.PZoNYg0IljNZ8zXwTX4B5ulAYvQ.ni1Jmjg9NMnJ6LPJWONMoIgRHhBBmU+tmqIS3xi.ft1+kQDHw2y71kDUi.37VnCXpgXes+b7nIJ.4L+OmAqMTni+IQuGvAcPGDNtialXLiYL8pa9s7k55Zbi2zMhewu35yqomDDM37r.18Z5S21INQbBmvIfccWm1aRX+lmxS+zOMl0rtHL3fMQjKjHRzrv3QD1bvDfoN0cF+Ey73G1mjaO3C9f3hu3YgU+huXaUDkzOwiR.Xe2u8CG+LO9g8Q93Vt0aAyd1yFqasM48glH6oDrcCTcBnIJ.09eAbcN7fkVqSeVei5Ds37VxRpGYE1vmCNhoPH4hPtSx5anYuWCcIOapSIw8xnHJ.bzKxFly3uc3+DbzvwCQDr9LuRwU9XtOkowXxOjneRzuZGAIy3OR.0oZL26et3Czmeff16YrWXLidj+a.XyuC.G4QOyuwduW6Y+95XgKZQsxlV.2feg6QgYpyk3r+ydpWzonBX51Kj8HTsK6QjuGNWjK554nUPJOKe9qAUvPg.RHT8FVM0RIvWoiNAhjlPuHTqyNHDcpnUIAusMdbdvKbmGpn5WOS.Z1heIi+VAc68cRX+2+g+auuK3BNe7TKbg.HJiYNRgX+m1+e.u+C.G0QM7e68cK2xMia7ltoPBMBDoeU.Tk9SngmTUI3vNzCCGxgbnCqSxs0rl0fq5m8yvbty4zd7aBwkYLWz6MpQOZbzetiFefOP+EN12pVdgW3EvYeNmMdnGj+p+oeuO7k9KAfJIuM5xF+8kIhzeQ5T8rZu4VojaDUWd.aEAT81NJXI6mo6S7052MlphqEys1f22oRUpwzbMTwyPuHg.4PvadDm7kd10+lwwLuRGMjDerBPSRAp1DhkHtR2MFsI.fL8e+y8222N.r0a8Dvzl1z1jTX22N.rkie7egce25+Cbn6et2WgA4nWh95vinA5dUpyq+d1kK65Tz4gnG7vmsG4bYyEcO6+lr+2dI5C.j5sV1poPYGNz1OK3Hp.NaLOi.TrzrP.wzeNDj185veoVkD+dZTFZF75NNXrQ.zL.Welcr+le19tu6KNoS5KhILgIrgZ02xWlyblCthq7JvZW6Z0Q3.v4Gtwv3tSYra4XwwcrGG1u8a+dSBy27TV9xWNt3K9hwhVzBM5zSBKoE8GjwRIL4IOYLyialC62deO0S8T3BuvK.KavACi32Xzu.foMsoiS3DNgg8Q939982GN2y4bvpW8pMCgUhtKr30gtoTmMZ2LaZOB.QdDrH.DWznlhG16TXc74n.Xqid4LiHCxlCZlg4lFQi9avwDJpBpA2Z0.amaGPDlsOGwg3JrlL7NQ3ulyABxIZXs19tNkt2ZfjSEpCFbNCjw+AFX.rt0sNLpQMpNqixxdsG6wHN3C9nlw+0+0U9v8y622N.rW64dLkMkui6O38+...j.FM6332BHZF8sKlm4gOfOtwJNf.lgdx3ljCcd67.H1ovduoI5m09ZapXY1XeqvzCFubmSb3iBUQePHGMR96aIMivulzh+oXmkDM4kpwCqlGAfTlQUl0riZziBG6m+Xwe9e9edq9ggSkgFZHbwW7rvCRyzoLat4vcmnnlr2yXuwLm4LG1e1Fb228cSauOKKRTOeBquZJeFUHPUTCbfG3G.e5O8mdXcRts90udbi2zMha7FtgXT3Rbpl4yvTuO.fTUgO9G+iiC+vN7g0Q93ke4WFW7kbw3Nt8aOGdbO+FpAbCrhjm0Lf4PTROLcx5fT2kX0mlwa8u71BjB6etcZMKXUuHDalzvqJq9b4TeFyldeqdSA8s0YcoIzDA25pbzN63.BxVtgjemDAu3MTyyzHN.A0RSzSqRttz1aGPpsHi87xbXIgIw+W65VKl+7F.626p+9Tpu2yXOwDlzV7sAvQzOueeYQ+vOpYdj61zlVeg..My7XwK9oMaZ71+yLXQyX1CqC4D.GNtrwY67n2f2lLiMy4Xj.LDv79z+2nCG1o9mdX4H7NJrz4DGW41GBYnlbNPJgofmHhKjwyJgC6Uf6zgirMO22y+1Q4q62sArNfG.1VcbW2kcAeou7WF6v1uCX3bYfAF.WzEcgMGjITzOh7uxn2.LpQNR7YNxiDGzvvOfObY0qd03xu7KCCLv.tC2k96pieHkq5z+F23FGNli9XF1u89V1xVFtvK5BwhxK8C.nAqNKIr00PCeYJSdx3DNguvv9He7XO1igexO8mfAW1f1r9qQiCPhcMx6nJ1vKmPfvyWHsTNnRsxmD3Y+uKi4yWwmws9nv98WHXXbQ3Yl2TCwylgRinMukmPf9ryq65CDjBO09tN5LQlXLfbPBI6.HBn4CDjkK.HNryf1VmeNm.bZ1Mx03Tw8e+ysuc.XG1gsCa2j11ORe8xnOc.XKFwH9W2y8r+y.7G39mad8ahIqWbM9YUxs6TLC5.s25e9DYhvqUI0uYxfVHO6pKIWpSYuDaFPn68e67.PntTgDPnP4qYsehbtQEln.WzTEz.C0bThE6bYOa4L5xhF4ygc50oKkAsIuxG1OM.xq2+m7S9ovQbDGwv9Lb+ptpeF9M29s67Cw4rM9CoLrrQfrvwTm5Ni+h+x+Rr8SY6dS.y27Ul+7mOtrK6RwPqdHR9wEf7Y6RJn0h.rO68dii9nOlg8I41cLmeKt5e1OCq6UVGfp+An4Zf3DSxCHSY9ze1e5eFNxi7HG1G4iq4ZuFb8W+0SFoZTZVmbEGkFVYa75zPZL9WqJfPSdEYujqPiMbC03NL9ewqX+F1r5KRPNROa4olWSwS.bcx4VaBOmlznMTAwNaW5pn3rUHbwhBKPHuBjrgbUQai...H.jDQAQEZE.ttc1R44bCY2hF+03KSJ7as7.2+8i0WWi94quqHUX22yceK2nuXtzWN.LscYp6+1r0aS+Vm39u+4FN6+4q0jkPmkawFNxpCe6PH113nxhDfO6acBNJL4p0LV6g+OU7LpjS6UNTeVFzKI+HvMKjZmZdpWi4mo6cdfDs9ZrWsQCO557TpPNU.S.+Q7RyAUEmURpxWmEamRz3DLFQUi2+SYamD9xe4uL18ce2wv4xhVzhv4c9mKV1xVlw2BmoBrlmTx5+..N7C+vwG8i9wF167yO+mec3Nuy6zDMpBJd8DXxcIzG0M5QMJ7o+zeZb.Gvv6s22pV0pvruzYiAFXfvTJ3YhpmY6ZFWqu23G23vwe7G+vpOcwcUd1m8YwO4m7iwBWzhr9+QXyVtYhM998uAFUSQy4AheVgnq6tsM5rHpUFtbnV6yWlLCuMSBpsg6vreSp9KUMqt6p7Yk2zWw5ynCTmbwz4mfcl..viEDaxjkKC.6qrecr871oIxDkF1GQVOaMRXDUBVecWv2fGN+iqWG+YXFZngvBexmDSuO2F169tMM7o9rGy+v08ytrSYi8taTG.l9zm9Dl9z209KCD.vKul0fG+wdLmHnYBCzHjoOyGkVDPddFZHu0NxFmMigEydgYlg2Ig7dMsb654vWAjCMT1sDoQD2WlBdxi7Lpz1A.onSMr.sNqbyImVyHScnfaCzB+8W.wqsVD.ojknjJlzrs+x3Vl++g9fePLyYN7e68cC2zMheYdlNla7VXdUGBSAYpTBXRSZawe4ewIfoMso8lD1u4o7zK9owkbwWBV1fKyT+Zq3uPKyAbdhI8KBl5NMUbbG2LG1mjaCLv.3Rl8kfUOzPtSfvGSpZ2CYycVdYe228CG6wdrC6i7wu9W+qwkdYWJV25VWVsj5nChzeBn7rN.Prk3l+Tg2n+v2K8R97EICRzIfL+L3Dd1hZTWmpmllDWv9nXSTiWqbcV27NFn7bDflfdywCuE597Dy5v3OW+w1hbzvLoT3XMsDGpoc6HjOve8VxV1Cmo3T.g+rSXOz.OXe6.vdr66FFyaaK9qAvqcG.lw68.9FaJyP7Qej4i0st0QF+gYjJaMyeYabYwgxitFLhNietPcmpAcsMzYPq56y791geOFA.+PhH6jQRibflnbdKylVr.1GbH.NcV3fSSmqiEIl9g6Pg9aEU4qKcBPCim.+H+U2WuZyWIvVmuwskaI9hmzWD+I+I+IX3bYvAGDm24cdXgK7o7te3I3H+grQXmsxu6dsmyXXsw+55Zbq25sfa5ltITmO21gH9IaJ79eQG+oynJAHUU3P9HGBNjC4PFVmjaqYMqAW80b03Nuy6zbpAcnzMZqpQevnF0nwQ8YOpg+GrQuvKfy87NWLv.CXFbCVlknpC97x2xdcw+X5vG.PptjlaH18AMCcqjMd2kQulmquujEGYC558Scz+AS7kIH9b.PuWboDb5kWmd.z42G.t8UVHOUwxHNnzuEoAz39scxC1iRXhIfcn.geyPLuGZ.bDGwQtAqWsL1wNVra69t0W6Y+MpC.ieTi4yN8ccW5qFF.XdCLOe1xpjGbi773SehtRq5wtms+M0Sdu7Zxmn2QMRpdcpyJ1b7fbVfKpPfcBCl2BdPr0LqpoocmJTCLfL3KjCAh.OhGjKAF7rGvtmlAdTd.ois5r6a67jym0iiR5y8KS+0IL4oLY7+5+4+q++EauuK+Jtb6fLQU766keoPYE6GZC++2NmeKdmuy24vxjca4Ke4X1y9RvBWzBsY5YxUYG8PP9yEfU4zTpFSe5SeXsw+EtvEhKZVWDV9xGzc1CvF+C.K41zj8SMXTIU3e7q80v1scCu+lVL2496w4c9mOV8pWsG0CZ6EY63Acl0.A8OMqOtd6jsa.zhJwvNBr9Tig1N80JARqGLC25xBz7JI687jvzmHCmzx5DbTZK.OIy6zkDljkGEjnob9y3qi5EoHtlvYbTTX9Wkx+X82wkestV8GqinNv5rPWQKvGVuvE9TXUCsJL9w0ee7wl1tryU60QbDu8G4Ztl+3F581ni92s8X22gMks+2.OzCByCM3BZkIbD5R3o4ki+ttgAOBQr0ZuJWGo7+i8Bjc9UE7ZzO1QB.pm6+.4D3PGzjqjTVvSJcdfB+uIznNgz.ah9Wu8cbyTDqUYpMo6N7pddPOvFXketcn+zPVo7eqS4Y+WIXvksLr3+vh6foO7nLzPCge7O9rvrl0Eg0s10pb2fwOn2Koyjno+oz4OA.y5hmEFZngdijDdMW9c+t6Fm7obxlw+tn+lhPzumGIo7iDHX1yd13kd4W5MAp30Vo4q22Mfu+O36ikO3fvmSZQglogleL5n75TB20cc2uAgwa9Ku7K+x37O+yC+vezOBuX13eSgzwQyjWWtC8wM57xulzLIKamAkem5rgdMOppqS1WgOSWq1.gh6zkYSlVNJXvmHvK1YVhUCz3WpFrIhouJYfWesThRFPwgspIuxZcf.oFlZwIEORyI6VF+Tx3pH9wmNPSjkUrpqI4FS5u1FCY9G.v7F3gZ8N8pL8oOMr2iXK+21Xu2FzAfC8POz8XWl5N0Fy6QYoKYIz4rtOCY9HdLodHZ2pzpWTQlsd1pGe4Y6HYXqx2ylzWRc3HOQPxnbq1xp+bSKYOlsYPzTIp3ru1T4eihNVweOi9smGOMDM4HJJF9fBVTTnouxsihd9w8KEPC9iZHsEeDb1+zeZyrEFlUFXfAv27e+ahG3Aef.OOr+9EINfK2OXNDTDwkUOzPX1yd1uASIu5Jqd0qFm+Eb93xu7KGuxq7JPcxthoe.3xevhXFKqwJidgU9B3pupq5MZR40TYYKaY36e5+.bS2zMRQLKWrwU5L3D24njOCT8ha8VuE7XO1i8FLE7Zu7DOwiiu9W+qi63NtiNk+AfS+nstlphwO1Z+iFcE9WITRlATRRKbxR2EFJ5+mU+Ek+zmGTloNz559Ly9zLvYa4tt.cYd8IbAn6hKPN3jr6kYFALmsUYlAxDfypcDk44.M7xJ3.53eWy1Mx+ZDUS105RWqMv7FXfNqitJ67T2I711pw9I1Xu2FzAfwMgs6+8tsIb5+Mu4MOqyMl0mdXa5hgF7ZkdEOgMZticHK.2g2lw14Y+S9NjxcPVFo1Q6nKuPcNJC9Ahg5Ig9m1GpCV8mTmZfYjI6xhgnVeHKDvNBYdUWtLIBUONiw7y.nYl+T3CpDAoLd0jIrvV++Z.rhUrBb9Wv4igKk0t10hK8RmMNiy3GgUsxUBmSwJWbgAIq0QYWMuGKTDUz7PC7f3Nuy67ML54US4gejGFm7I+cw.C7PjxwXRtQpAc4ckOPJg02Ru+bm6bwbu+49FEo7Zpbm24chS9T9t3O7zOcftRAm6bGr4wjMun4Vuc+K5htP7Ruzvinfr90udbMWy0f+yuy2AKaYK0btgm6uFsGWcQ7LuPKZtMoKupEFaQLCkl9MEFcsty5yCpuKcDHyeY7Hp+yGg5WlLmBz64SGBA42l+joe8cy5Tafuoy11B2I3e1gaHlNyC.ehjDB.yGxXILtxcNwbznpJ7ZELHx4A5fqScjioobEL+4MuNqotJibjiD6wzl9Fc8s1f68o26G3O8L9zehO1V1uqS3MbC+Brjk7rjmeB3YCqyHoovKKPziTsjRHOE1FnXOWA6gj3BIkyHWeG2eLpH99gWguxD9sJHh+pPYlHa29kc5JcCXHhZGpS3cAoLzA5I.ODndiWUI1w2ow+fFM.gbrQve7O9GvjlxjwNO0oh2JWV3hVH9gm9oi4+vOLzY3D7Eh3qljTwTRD8dD7H.ufGcAK.6+6d+wX6yu5VuQUV25VGtlq8ZvO+5tNr10sNW4lpVTUVwikT4m7+HlbhX7Ae1vM26wVvBv9u++IukcGgLzPCgK7Bu.ba+laC0qu1M93L.a7sMNBtyeg2UU5lgaMqYMXEq34w9seuq2XIpMwxRVxRvoe5+.76tm6wb3wjsAH4evycIve7YTlhiCPSdSk.sk.Aergmc2TxQYrV4+UbkTTxNjY+T6q7nzPFJhuWXxW96pxr7YDPa3QA7MOIIZxPmqAo7aKCWAtk8Viwn5WZARxnOATx.VE6qXlVP+NYr2YGN9r10tV7ddOuWL9w2e4Avy87qPV4JjYuvEN+A606rAsruaSaWmT+t+nSoZrfG8QIOkbBzWQbeUu6ZMQzAmI.acZTu2JdMC9D0IWFZXd1Ps7dK68Wyr+QC9IMdLFweBy4Y6mqX62P83k2e+dXoR4WQ6jkB30jgIx+HoLwg2cXHAjpMi+gv0YB74DMIudv0Yg+K9hZRdp2JVpqqwu7F9k369c9NXoKcol62D4aiTByDQzcdhNSh78H22MW6L3AV2ZWKtnK5B2nYu6ajkEu3EiS8zNUbW1d62+W0YTS9IqT2Td5dnVL9KWCjCuIjvZVyZvkdoytmgo7Myx7l27v2467eh4M+46jF6vSxivmoKWMPlEBr6Aeu+yId08de2G98+9e+a7DWeVtsa61vW+a70a9fVQ3eidM1niJlSNH.PQIfRlMR+otzmhH1IEXRZVdUMJh43J5ei.pZ99.Xi+3n.vdinNbkG75qoe92ZDRsNvDoiEzXZ3ISHPr+WoeV72bvnowkjeJHB3KMP6cB.rj4Sq.sc88tuFkADoezjf55RtVoJgqaj65bc9M8RzVzLE2Ub12AAjvC+vyuCIjtK61tMMrUSYK9eugdmd5.vAcLGy9L0oN0N8sqqxhW7h6bsk8szFst3.8TYi1m4mYyMFlKlfia4jcXqcq6gEpzgC8fEpRnL+G1oGkIeQXNjdz9kHg9SJIU1Xv2xaRWphQDym.tlA7s8Whpj5DZB8Ud1e7Qk7K9xuL9I+je5aoL5Azr89NkS8TvO+5tNKjiZGho3Hyac9J22JQEOfNEJo+F3dhfEtnEge0u5W8FGg1iRccMt4a4lwO7Gd5XvAWVg7S4rPzna47GU.wDy.MxSbmDKkKehm3Ivse629a7DbOJqcsqEW9Ub43bN2yAu3K9hw9OvNBpzG7YLJ9u06oSHnR4eoT3cuzK6RwJVwJdilL2fkUtxUhevO3GfK9RtX7JqacF91XThvep+lUCYSlRb8KByaxvZQAM4WWglP9qedfSYW.rn.j.fXeiT6VGrl6EJdIHz+4fQxrpiaPn9OOZW5LhcG+ix6gqcFPfNA7SJP.zJO.3Jvxo.ZLj9D89Lwyw+sVyBaitZaGxGSJgeGeGUGmfG4QdjNXzcW1o29aGiaricCdr.2SG.115Q8urqaBmE1O5i7HvLXUZ7JW3SuoNyJexKoZ5j4i2Cm9ZRI9.daV.9HdJ+P0JO1V5oKHP16WSdEMCvH6JRG0hPsORl2mMud9Y1TRxOLQ3u1fPElXAd3yniQayKZ.I+Y9EYOz01v4povxYXmA3p2oPvirfGE2vMdC3sJk4Lm4fu427afm7IeR.3CDXiYFOxLvib+u2IIk2ylEh0ASNKASI0Mbi2.VzhVzqGjVeUV9xWNNiy3Gga5FuQTWW6F1DhCX3uaFOlmKHpLgg2THR7OqZDbC2vuDO6y9ruNSka7xhVzhvodpmBt669tZtAaMWoeq+2W6T8sr91x9e.eVbjSfI.rlW9kwEewy5sLQA49u+6G+e9+7u0j2G4Rh9GV9WWR.64Bo+AkZA8cBh5fseVmjB4FfTIVBApl4rCFHAv+BAVDE.y6Rm+6H.I+FzakW2aJ2.DxqNSzMOaX68gCyFBdMQ.0HgVAw99uDOrC7wOdTT7mYyIwveu+P4kJ4Z1up7Qqcdv.kbbMEpS50x2+QezGAquOmz1HFwHvz10cYxan2omN.L5wLpCYm1o2de0P..O77mO.z0Y08Jhy7cQHOvQjwFdm7GkGcq4YetZK7VhfLDZduSyrlZ3S2k7I9j3sCx1mciwLN2U6SdDxvaBgL96vIJsovmGvEQ.+V9D.cAI8Xuz8P2SjG0iW0yT2PYiyOW0UcUuoZzCnYMdOqe7YgKZVWjcHRAQBJ1T8It2v4R9AgCSC8cnLoobFjJnfaqTByZVyp4yG7avk69d9c3zNsSEK5oWjSSFAkxx0J96BEp7iJqY7GxvtK+wQPvFgZ7p0WWiK4RmcdWF7FeY8qe83ltoapI5GKe4FZqFyL5WTiF1vByIZe2e.vUfqGpxjezgZBZTL+nKXA31tsa6MZxNTd4W9kwEcQWHNyy5LwPqd0AinlM0BYXU+gdsRWV30gAn8bAtNglIXkrHDpQ.PJLlpldCSXVpfHUtsbXfPNi36.KV+CQ.1LpIuVCKKPDdR.lf2lMcK3alvSSzcqb34sJUWGJPrCJV2fz50L8+.VdVoKmLexCJcAOzwuIu+gwe5c..doW7EwhWb+uUtm5NtCxAevGbOOlO64B7eDG4m6+789d1+MjUSqjR0X1WxE2bDTRHavfnYCtTPHWnAk55fTIUY4AuBTAwRkgwq85raC1MOqxDFZuchLcGlBxvi.GNnXip3KS+uZgWb9VK3a7vTGmYelOAo+OOPlo+DwmkriCKXAK.GzAcPuobd3Ov.CfS+z+A3oe5m1tm28QJ8.oHKQg0mLvG5+x+kS7sf7GhJirkF4EWMdwW5EeC6.BZ0qd03Rl8kfa6+5+Bqe8qGFxAh9CtL5+sLmWP3sbCgN7oPMzE8u5gFBux5VG1i8nuNHw1rUFbvAw4dtmC98ykVKdc7aVnmiXCKXnimf8qL3fn+NlYuP0qR+KXAOJ128a+56CbkMmkm3IdB78+Ae+lv7x4wB.3kSDjwB+mLu.QmdK3MVjhxNGoew+ZrA2ztMepfEJOhbbgc1TWeZGO0JGjdeec7U6pV2mFQifTd9dr8CjBOqoITECzgDjy.ZAesweb3S.cmG.lCIMuokaETwivaJLVR+iI8ItMstJ9tXgeGdBgQX2gcXGvzld+s67dkWoFC7nO9Hej48f2XWOuyH.rO6y9Ltc7suC88QD1y7LOCdwW7EseyacMKorD2nm387VgkW.7vm3LNRwEKP2gsc2PA.m3gMUrPd7Urm+QNwPRtGufw+rTaPYq5YcAd4uRxhNfJLozu6WH83R5JnwWg2OvepDc+5F7gv+xIlfk3e.ZdNDUV7Gel+Hthq3JZyHecrr10tVbIW5rwO5G8CwpV0pZ6lFwiTEbtCA7.e3Dew8Mkk1.LV9iMGFgeNy42hAle+ukad0VdjG4Qvobpmb9S2Ki+txMahTJtVH+4QFCsj+jVvGo+b007Tp4+M29siG+Id7W2n6xxcc22ENsS6TwSu3EGbhsgR79O0ngTEoeiun2rvfXxsXk4e7qEge80qGW3EbAugFEj0u90iq65tNbxm72ECtrAUByvQl989+jasSbYCltzAOk6ZhfoJcRBDeAf+fqkPpVyC.cSYI1gLl6rcEHvikTxk8TbCLYZDfgyMNM.Kxbt4TpMMGOH3U9EZCuliWIB9tyCfrCEJ+im7PWi+rwOtVL1g.coF3bQhKAGUD+lhRJrCFBvi+38+XycYm2ILhQT8o50y6zH+zmw9+01kcdm56F4werE36Qe0iyDcM4AUaC5ME+i9fcG6pFZm7LJafzGbPlSUYgjecv2xrjaU9L.PC8u8kuh6D0dDg1OrYgBe7Ut8UOg6.9VQ6fVeZN6+U9mKHW7e5MCm5eYg8TxbHf4ahnediAcRBmYNIOiau4a9Wg4sIrOSesTVzhVD9Veq+c7atsayTBEmUR9ZkCptP6XOoriOIHSwAXk+0p.xia8wzZKKPvkM6Y+51oD35V25vUe0WEN6y4rwpV0PsZeE2bElY.McJQ4GAQ4GNBRA30hS9NqH.OvkcYW5q66O9gFZHbtm24he1UdkXsTzCIcfvizG0+yNHkbYYEdteOXvQYRj+0IT.OD7GelmAW+u35echpikkrjkfu6286fewun4CZE6bS.+Kj+kR4eCjH+CnglJm6oqJnQ+QcJ4K0ZtlsDBTjvjjE1fTJ0nKJmO.D3jmmBbMzda5N2qWPYeu1G2GvWJ+pCbzvoWFcL87dA.9WY1PT.jPcEfOa+RsIE9Hjw3Occq5uG4APflS45gpOs8e7GaAneKiYLiASaWlVOCWPmN.L5QhibpaB6Q7G+wdbTAtmG901XQ06mNTHkYzJjpwYEtrnEYbjGXGm8coaDgq407A55BkgSGik6bBg7gWeepNCy5uzgjR3wlF79HZ+u9m6Ww1e+pm4ZPvTGBzC9G.MB.s6rs0aJy+O6y902SIv55Z7K+k+R7e9c9OwRW5RCJq.X5uPgGOCHPrjfyVMUfoOgqKPacn7Kn+1ftPIypFZHbYW9ktYlCzraYNsu2og4bmywGSTz9N9CXdZxJhBNHHF83vP+nS3a9Gd8FKgYkuvJwUe0W8lc5WKye9yGm5odJMaqIUOmx+MuqgOtfF+anbgwdyf.TFQxtevQ.gfGEvmwua8Vd8+TB72b6+F7s9V+6M4fi3yXsQV1rnY7jRGABF6UXb6Djy0EKm.CSVmrpaPOw7TcGZxAVIBjJOgiEjmrAoL2aSpwHq7tiHj9Ow+U.dVqVWx+jAdu8on8Ry72XY5XrLLIQBmDrwh19v9KK+38ID9m7tG9Z+iMWaGwrBQyw126g01+EdgUfku7k2qZpUYG1gcnmmk+ctnue3OxgeJe3C9C12eBfuhK+x7spC48j0sKZRdH4wjEL8rfsHRCSpNAopxb9ySfml5ohbVfmgfO6mTn8sh0Ozbu57ysvaowRGHHrIg6wy9xEhsmSzHS+98JweTvWH7MqTV8vU0QOhpJ679WUfaBnUMCy31uh3ulW9DBnJVV6ZWKFbvkg26688tQ6y2TKCN3f3GcF+HbW2kep6UpXlWudG+89Kk98PfhV8Ow2I+bDyT7VKAf4jfKsTIBV5xVJlvD1ZrS6T+GMrdUpqqws9quUbIy9R7kKSwennpKKENheKj6zG43OASo7KO9fT7x0Pn8owuKYoOKlzjlL19sey2GLm0t10hq4ZtZ7yu9quIYKYw8BdQFYc5Niy7Z2y2Wo+F1pej1DL7yQOIQvmRs3eO5i9n3.NfC.iZT8spv9prxUtRb1myYia4Vtkv1vMh+tw9R8kN9GcRPmfk8NpytrtvTT+Und0deND7JdwxOYiUwkNON9o.igqHikeY2AH82E1HDgvIReX+CuecJo6fA0gASr.lgfbs5Wx6XD2oj3tXKQ7+Tq129JMlRs5Oyf65+C1OTyCJ+u4t67tryXm1o9aR5uvJWIV2ZjG7QezAZcHBzYD.1wcbG1h9plAvJW4KfAGTOPY78MIKvXmE.jQPt3JgQN77U48uN7YkSLAeI0kPsj.rtTW0VgXNkUldTKD6q9mKm5NrvCdTgPTz9hAuGdnfgF5dF9KNU3xDD9RdS5vWYJMTiDtbaxNDOb5KGle6cEiepxqJ+OAf64duWLm6bNXyY4NlyuEeyu42DO4S9jlfMT5mzM2vrYCcr7CAiNiBxfAqvpblxrSCr7mEQJBC35Rffq9puZR99UW44dtmCm4YdF3FugaHtllJ9KTqmn1WTUNp7mq7ybLPxOyDfZmw0rY9nhEt8KkUad9Uc0W0ls8G+S+zOM9de+SC20ce2Ex+NCnr8iioP1YtjQ+J96UPxjQ3vYGcLuj+6FdY3dgW3EvUbEW9lEZWKOvC7.3a7M+FXfAFnS4TTJqWbOWoHYv.lIV5cX3Kj+SrDAK4mWhVQxmI.MOgOG80lW2wQ.Z.UYYsLtH7+IFcng8OZHzMppc67XZP8oF7negOYDoNIHcGeYIIcGQBPHgR85JhOTIbzmK40HH+YGKvF+uns5vgEkV0k7R62pQBOwlPd.ry6zNBYzi7j55Ysb.3POzCcO1tMAO9epm7IIBx8.yowDB8SHgRVsAetSnw.GumHkVvXhYhBOGlKc5AsNoAn0hIqbMCeMf2G5fq+Ci9cz9j.O37EHQ.vFXx+SpC34YqT991uqLAwDbAJKC+sv+mLgc.XI3nxOU+mSjBEA.y5htnMoPL0qxPCMDNyy5LwrtvKDqYsqAHQ8GlBgXzbPd.hy+f6oRAKw3YE3Oex+ouhzvNLZ1UjRCHE99Mk0tt0hK9Rl0q5CLo69d9c3TO0SAKbQKxoAg4+DYZFxYYNdYZTb2kyMCh1i3yGAmr.qfQuYVQJGhz.4mZNpbu7q7x6ToU+VpqqwMey+J7iNieHV9xeNq8ciEZuh29thekeEGPzzelh3uQWpmyEueOg2jJBxeI.bu268tY4TBbMqYMXVyZV3G+iOKr5UuZp8k1sOH5mv43OUmDQgCA5uy+mulHw5z69MCKZ+uexq5X.anvGulriFXEmLzoj.hhzFdFcHnGxuFHttwf7eWvm5BduzrMGadZccc9qNaGEkIIkjjOQvvGPHS7KkyKNsBhknyOw2Ko19XSGppAowwrm5odptv1NKSY6lB1hsbru+tdVKZdrSXJ+cScG2g9txepE9TgYpoEi9Dmnx2nkQYyS77r+qppr8jJGtNNb6Lb56YdJBEtBEG4yW.FdEjpbOrPBRMWqJqS1.DjoHVIpGg.82vqLi9UEOa.3aXQAAOF9JZDlGTHw9p+oeLOrS2qb62b.e3gCN3F.a3TDrl0rVbNmyY+Z5TBbfAF.e8u9+G7.2+8mIfTn+T4+lXeVQH2eV1+q3ICuo3hveqei7aHU19EQZfC4p0N46unEsHby27l1oD3pW8pwEbgW.txq3JZB0cwrtb7O+WgWqVt8MBn2vqiKXdTA+SM1vg+u0Ru.OasY3exm3IvseGu5Nk.W9xGDmwYbF3Wcy2r5ub61O++7YlGGG1f9zfyV7uRYI6lD0FgOYv6xIo.ntbzq0SIvm7IeR7s9O92wcly6iV3O71m+s0+EjMTBhbdVeGKRANCLFEg1i+bFHbGhEj+OwLP3ItsP3KCu.M4j49ufUyMT+m1FBS+b+OwX.2+kJQet6uS3siE3bcUkO14qKNPfLbLQ7Oib3yF.DQfbePEDJIBNDNK...B.IQTPTYCqC0czg5lJHP+FOTHdniA+gEu39dmpLhpJLscYpcdf.01omQHG5Nti6XeUw..K7odJ3xAA0SM+Ia3Njc5HxjyulUZNOkUlje.OvtQ..K7H50RnhJbAUO6+g2ImRf9BQ4uqdkW+h0QXud1gAmdJUP6cb9.uTA7oVvGPaCd.cMMKyFWndgy2WwqTyRBTAPQLvaQX+psepO5BV.twarysN5Frr10tVbIWxkfS+Gd5XkqZUlBJyAMJbl9rQDy.kc+TxlIlpLzhf.aTSmEQA9mGyZ0eq1WjNaeqNzAz4A9+pa9lCmUAanxi7HOBNkS4jw.OzC40aUTFiEVUpNLtPetJ+Df2sXY7iNlngNOVFdU9yWJFVVP5I7+pa5l1jOk.u6e2ciu2266gE+GVL0dZyv3CkuLFawGu4JBY9EyMkfbkS.42mLDnvabSRosab0+aBu5Ok.W+5WOt9q+miS8zNUr7k+b1P6.9Sx8j8sL5SSaRhIonacUEU5l9CiePJPu7y4v9v5O.hSbPw6FdJsyiRIf7mlbEOawtrPc5F7UJP0yp7bG9rjg1FA3QA7Xi.e94he5xZG1bpBzPhMj2BkYKxbNC3sIo2tb7Sx4kUUUEJo6XrF4biytzIt3iGADr90ud7G1DNPf1tsaJUnib9q0MNrC+S7c+f+Yefdl0fkkK6RmMVW9jSyERbOMc5TBO2tKI7qdgUYy9.16FutM7c29DSNgP1+2Fl7frL5w3uYZWcz4UM7RA7Rgbf3+QuzfOO.Kymh7znRR9qaXRhyxFt9lt4e1rnErfG8Qw9seuKLgILAzOkEsnEgu22+6g4Mul02jMbGMt169ufBKIXRLnnTuXCg+BAS3xNkYXihT6SNN7DO4Sf226680yCLo0st0gq85tVbsW601bnXIQbNt1fM+mi+da4Frj17DlmIEIKnZVSDqNzYIpvGpGiN012UnVBecJgEsvEh2y648fM1WGzgV8PX1y9Rvse62tsbdLdyzLmfmtdTw9M08PzMMVjdW2oNpu2IJ2QmdH+YvK7HzFvW9y8bXLawXvttq65Fj10xRW5RwYdVmIt2669Z43xFr8KhBEK+W1+0a7GsvetcaqSFlrttTn7WI0ZxQMIaPrN45zz3SzLQjBZU+gMVfW1Oxlf0UUBeQN6.R+YP9WoKOJZbaxzsqaDXDheckT0Zl4Nt183Oon8SDs.H1WePEtF0SRKuiLb17CvoUeIhj.X6zTmJ10ccZneJqdnUgU+Ji3wVv7evGfueqQxa+1Mo2VeUi.X4CNHV8PqBDVCWoUy8LO6.nvH4E9qjjsmLg6QXcndbp2pe0uLQetDDjBExCLcqwwe8+7PHoFRz+joGpJCs+FE9Lt0I7A+38QrpikLq0mRqsK.7u2D4bOPDK4+DQrH.XvqBl.YENvCcHYnNkZNZX+o+zehcBO1qRy166Wfu829+v1detm4tBWG6gOiEMLKACAd3LU7DLrb86rp7KHAZILyHq+or885zTvZsuC+fCtLbcW201IOXwKdw3688NMbmyYND7pBuLcvsOhQBwTdId3vcm+5B+U4xXcYgvmoItgfqnwdQSoDHGIaC+ytjkfa5l1vQE5ge3GFm1ocZX9O7CSJjEhA.ihb9h1Wosux2iLfn7S.8sZSweCrVxeT71L9RTVwZetdPBW20ds3YdlmYCR+..2wcbG3a+s+OvS+zKpi12o61zOsLU46EnCV6uprQ0eRFGc5igGf0eX5BnnLjR9Vit4Z+ilij.ka.7B13QkrttIIk0iF3R4NqsKX.LuPo+vXBD4et8.c3aotBeISrk0nCxWerRi0QD0PdSNLD8RO+E3qMml0wZYcx52hfdWx8eYAPU9C1eyukJefDV3BepMXMxkse62AL5Qfit79AG.N3C9fmzj11M32NfPYgKZgv8cwUZEXZj893RDzq64+toiQ85M+OhZvJ++D6Jp97e2PkMWqedgQtd0SEuLH4V2UkZJYDkwyJus4n0Gv2nUSgmMpDo9nSBN+qoAppburKNRCxI5m+oMFngOkTMpFupn8I7iwGcvyy7rOKthe1UhdUFbvAw28j+t3ZtlqI+Argoqj0lMMiXF2c2wKLFP22LHm+saHLQ0o+7VgAUaGQb5TUcY2WHixb6qreweWH3ttq6Bye99toottF2xsbK3G9COcrzksrd.OYXkTcpDfP.vyRw8WywUS0qIWBDH.p8MChffmbNW.8L3saWvq3+c7auC7DOwSzRNXsqcs3ptpqBW3EdAMaI3FvM5W4eg9ISzjGspbCd13JOzkeoAlDTJ9GMHvsUT9urOHy+LGp0s1Udlv003BunKrmq85KrpUhy5rNSboW1kh0st0ks8P7Vq8IYZStrr+qPd1jU408W+K6zqWU7WLPk+oFW5Z7GfCuHv1xvIU2k.+aLBzSfT8LIoANMGkL9uV4rm5IR9yDcEu4ceSr9HWOPDdzE7nGvK9CrkAPnufe1DP4BMdPG+l71Lf+1eLWpr52NyYJNnm3hoWPctmWdJiA.69KbS3a3xTl7jvHF4n1+x6Gb.XrayN9k2gseJ8ckt3E+z1r2UixkNpFLD2Qg+1G2DZwFIEQDy3rJ6nW3KOPxDFrObLTDErR1E0Zn6v.wDvqMltNiKJhBpDVlnL7HfWa5vqNt.521nCENU+FM5QmsutDIJoUIhk.fpmlbx.1bJek7IaCXBe1uoYb3JXZZ8a8VtkNOk.myblC95e8uNdR0fPO3+d+WxZmPxIIsmoiy9xCDT4r78r0DkTpH8.+avIGSL9eFWUiM75jZtPJs6+trK+RwPCMDdtm64vYbl+Hbi23Mf0WW6viR3A4aGkncDeWk40.RvFGYIsR4GjnWwveUtgbXnGvaruL8y3X.dk+k.t7q3xCmRfKdwOM9Am9O.2y876L3MKFlAURZPJvep8MFk3N9asexSdUSqhMtCV+Xa3cgPe7aT92MJRqyKIip7t+3e7Ofq+5aeJA9fO3Chu8+w2BCLu44x3DuUoU8ZW9KN9yoQX7OcrSCKHlLfVjxz9+NfOL1whxAM9ik+SsmIrtqh7uldhcn.YaMsjuDtgnuRp1HD0ZSEdKhWIpGfMXmb7mgGuJguBRfdJ+v8vEW9SsynxFbzeS9vOuCvhlRMx4el9gAh5WbVkZzO47N0lRQ6KBvy9LOCVeelHfibjiD63TmZqr6O3.vXFk7w19cn+2A.+gE+GrYUvqEho7BHRLkEI7hHJo3gmWUE0Tu9.W5IjRvdzVYhk+FCXDuX+Sv9KXgIRknBurY.do.dUPqE7Rkc5+Y3ujOC.LgMD9OMaWqfjcZP4eh0FslABWHkIm64dt1oD3PCMDNyy7Lv4eAmey16qbFLE8qNcJlhQOqtYVnD9Oa.KUekFtbYiN3+J2KS2bjhB0q3uuI7ZFNRA7VDAqd0qF+zy4rwodpmBVzBWDAO0OtAf2sJ3ssGAK521Pm7+S5E7d8GT7YTTOfWwa0gZiNjVvCk+IBV0JWIt1qsIhO2xsdK3LOqyBO2ysbl5I9YJV+Z+e.W6JB.lMeu+U0yHd+uR+AIXoDdDf2aeW92qCOhAdGfaLWffa8Weq3wxGGqqYMqAyd1yF+zy9mhUu5Wz4mN0Wz9Ntsa61tiielyr03EqPNC2k7ugsT8FjCTF.b7W6Hh6X.R9UnubnIXIIc44nu98A.nYbpmVRksuRKHHyB3zkTz9N+mpSR9ygmocgfW5K30rzWD+q3WqPqhX8wNNwi0K8yUfazOIdD.zHLjr9hXQXh2rEDaeI270qe8aRIl6NLkssUt8Exlo26AbfemO9G8PGaWgpuqxUckWoMSftRLEkoXeooZ0y5uq48XtCLYcJ563LbUQVit01uSq1IeR4vIDWJiSUDiU6SYkwg5lLxf.M8ZGdWYHyaz+1bVamfu1XIlWjaC+qcEL9f6YeWsewVAywjL79yVyZdYL3xGDiYLiAeuu+2q43KkqSsM35pi5t062A7Aik7eo5vnM6VcjrNnGskgGrLUw.9.+ur8AFZUqBux5Wen8aJoMN7aj1W0.xiiBEoD9Tj9KjMZUAJ7AcXthIVbkgWaQfljb69ev6GyadCDn4DUWg+RZGs1PQ.J5SLuQU1EXbjLl4DAIG6DXI7Iuw6r80Gk7wlZ8PzlVVvBV.1tsa6vO8m9Svi7nOZtI349xsuKafDfTUgO8m5SgYNyYhcdm2ELzPCgEsvEFb31F2Ffmne8ZabCOi3BvXdqAOsTbaD30+1Li4LcTQqyMZVRfj4wUUl9KwChAVhSBE8FiuCRGYhdtBO0mp8edkZxc8DdO7OMa6Qj77Di6rCEV9ym0O2G44vPbRe.H7oAtLB..I5VoP2tSpIRrMgca21s99DAbvAeN7hCsxY8DOwS7b58BdDriu8cZa7ya9Mb4ke4WFC9bCZaQjV6yZabczPcuJbB.pdzUC8XyU81KpnBj.l29zCrJ2szqNUTAfZs9X6Ch2cYJyX72TN.+KI3lA34oKnupCueRRUr5ZM2iLv03Uqi.14FPOLzp+0Ou8Ei8YKOS9Y228ceMGJJEypEz6DT3wJX5xHXtHTa0E7lcpb+kNPVeW1TXK9edvej+SJ45D+y+C29lXsD49420BNciP.fMPU46L7ZunSuA5O17w5gn0H75qTXrfrlGHK.yY6FNjh+jrYwxS3sXCs8bK+4hvy3Ozn24JCs9HZcNhzOQgIZspSQEs91ST4UtxbggmTvqg2VUbVp.ILiXf7m.WJi0EiRPJkvJVwJvYclmo2mjJn+Dn126u29se6wIbBm.3u0JG4m4Hwi7HOBVxRVhAujnj6K4i5UcsQgpNtm5LR43d1vaWzu35WqCFHETk+cMZhNfVOoLOoRb3qQERoZSdqkZK95P6qne6DIW3eGfGaBvKV+u+MAnA9J.5vnKpisYRo4wHjiZJKqlZ+D.DU9CdhrKolHP26yWEw69.rwDt8yrbs33+h+CKFGPOpsxx1MkofwOgI+EAv+S8dAq8a6DmXe+I.dwK9oCexH8PySdPYbnl6WlY9g0r09..kr0l2R.PPbkB3slhtNTBe9e0RxBMdFoImz00ZysCnBQpgGRcIxiz23viML7Dpw9BXFZ00+mMfZyPLo4LQjCnNjz0LHX3YCz75sZvJDCH4dRKbcx7eBdI.eJfg8r+SeW0CZi2Vz+R3La3wpSiGSxYaT5ORSV6m31O0BdKDxIpVTGRaAe.8AOC1jeKyQXFA5t8oqI0xZ6GbZN0Fdk+yxVsg2G.31ZZ2+wEygXw4aQ5m5+39BGC8w.k88DCLz9L7lCsdtUzu7e0nqYTOoqCaa4Gu8Slraq1G.enO3GD+S+S+Sn7Cs1nG8nwIchmXiS9IpelbHVkc6h+o3eYOfJ5ouSXLPJ0BdkVz2wC6O5HO.Rv2Bm901tRJU6suJ9Z2nAdWudLJKZOnM9MCC+rW8vmr2xhvbxuttN0Y15GyC.kdTadl6E1yrdJw+ZrVSXGP2iYXEE1B2n3OMzFnIx5aJmE.SdxSBnp5OiumYv+.N3OyttMa61z2U1y7G+iYpftooOJ4dNZFHEzVD0kI7uLeR3bnNk75SUlx55rqC14ngIzm+WUXVOqqgYESr+SE9MkV1fE9UsG5H0FCdrwfmXHN6CHUCjp806GrneNK+I30AnUfN5eCsOwn7vVDUVTpPV6+LE5h4TiBUI79fCZlL4+yR.JEcZAOLid1fJE1h2yZcyQC0fhh9Q5WBzui.heYbvcn8o5vZeIPaF4qlMScf+L7Z8FjeIiKIuGWP1fLEkt.7FmwMHjH9g2+UwMDwCIc0kvCE+89+R34werxY1YlH8Sx+L9mn1WUHZFCKTbpWy5WH8BQ8OzX.FbxPqAiRilbp6LKg8c29vYOoTBie7iGek+6+2wQezGSO+vBM0oty3S8I+j93ei9AM9qcDxjxJREXkBmd6h9Kh5QI7rAeMgh8CEnlRSDZUm7X8JzzQTi+FB0d7i1OX.w8EIP3er8CviNf2G.1Q+exx2gDCewr+Sr7LK+wzR43uf7ehVRaIleAENA35axsLg+J3b6+GT6v8QYq25If21a6ssm78LMASYJi83mxj119txZR9fjIrB3JG4A5dGVjo13Ud1OoJwBKhHvBMioWMasQElTOtTuhZdNOq5hgE47KPWCKIaftFdXtBeO4cUZP0t5ClPv.eZiBO5K36p3vWEh.fjUnWm.R47lf+7IqUX0Fp8QA9yvKa.7GN7MUmqrgg26+gYvWgMLi67fU6T+qXvsoWfl4ko3MGUIA9yYGMMbgvuT9ecaUTXd6gCAhXOzqDyAEt8SQGLL3ClLJpJe2yXikLvUdYwxNTDwFC9B5WULZg+tC38qyidLkoQ3KUIxJ3.0+F3+YFPY6G5m0wBD9qJ+LGNHkrVShR3I4GR+gN9sfA.c.nN7ikeM4Nx7nYHUeOhW1p8o2Y+d2ua7u9+3eEyXF6M1XkC8POLraSe2H4G5u7r+YC7s3+NbcN9KS9t7CO9xcxAnw3dkVOdi2LDNyS0sjrFB9vmHXw+htVp9ub7iFVdeF6fzivvmbGpCvq77B3kXeRI8qGIvgSrOc1VUk5AU1uNACcxMJBvKikK+40edPd1Ait1wTF9qNWX0Wx.miX1K77OOd4W9kQ+TFQUE1w29aeR78pnG9gmxT5+y.fk7rOaXvVCAoDIcsI6U1I3dkqq8hl0+5gxfBmEAfPcjbMzHW+YlSqRNG.3u9eMDOcOEOSE3u+Cp84AXtPpaDwMznOHXHfGfxJGktfuIrf5dtMKl075UBjJOCcUGbbGB5p8cqOkgMjUxYaGkBk6AkfaL3YkSJsRjZbMuyJ4HGELi14ArQtOYrQU5T.iuV65qTTCAmcHdtJyo.lkCKUjX895sSN2M3PAH9GIuZNqITyDqA69A7mTLJkv2FArJnM7BwWUEpvggf2Z+REbJ7ZaV5Po9.s8IVHWeQeDDpt5H4F82RQBtKffW6dyzA0+Gj+LfSt7qxKDO2EZ2+Wz9IGlQ+1da33O9iG+Uew+aXK2xsD8SoppBmvI7EvnG8nCQtp03mB8GJdj.wCYZwEfHDGgwTMvSKSaA7ZjSA.s14heNj.oY6FaieahbIfmbzcL.lDG5U6yvuAvefML7na340711pd.7IrFggNeW+q6vKuEjY4GexYoR4mRGSxTQlKmuljYIcUpyFhHXIaB6DfItsaSHu+L5ejUibFSba5+k.3Ye1mw7ZSKgY+J9rpDgxXSpXm.f.4uFShY7x+1AnIhl9qxvqC58HENLQFxA.O7vZmswhqHCBjvhDDbJBurUqjBsDqRhUtmgmoe13O8WF9p7dts499g.TpNYmW29g1QFFoWsOZg+93Ah963ilT4ua48JYH16+iKmfxmRsfO03PSG7elVTErEC2s1QeyR5GFqNluFb8VJ+Fo2DTCjsVuT58jPcojf24lRL+KP91mWTSlmokv3rVnuAu2+Qse3vMI0FdythN6+Ta38gOsweB9Viefy+R8.dguOWWF9R3eJx+4B2+ygIki.I6SWHekZg+U1CBQ.gjma09UMs+zl1zv+x+x+Bd+u+9M8r7xjlzjvm+y+4KjeExwDDzezK4+H+GsgOsoAOqmQ2hwo5TPujPND3GWzTBbBX5Ga09nWsep+veCdIHi3x6oLLN7MmOJQ3qfziCDHRVIL9wwfVse9Q1msck+UUjsVA8qtNBVeEM0F6cRoFGVd1mcie5TpkINwIhoO8oams6lV9c3su8aWuNiyKKqcsqEO2y8bvEE0RVkrpDQ8UIwdx3EgXLV3VRwuLelgdpN0P.ZJLDP3BODooTWSYQu3dDxQDPEJz1kQfj9F5LRRAnf05MZUJfmxXzR3KTrYsudi71+S2DCgsaCTE6NOTWdiJps13suzF+2XviH7B+bRAIqnjGX2xwC8eST6B1vcJ.uEtRtsc88FN1Jz9V+X98Ym7nNBwaLh98A117jMwO2vAa7mM7YFVEeW.DgGsf2qyHOUmQj18wsuMtqvPqWkN861aIIYR4UDdhlnYj5vWz9L92I7jBS5cQItlq.9do.7b+OyqKZe1wpj18qc.L86x8c4DRn80lI+2JoBehO4mD+c+e82gsch8+xoVVNvC3.w65c8tM4XUlVGaxi+L7NnqJNNnS3I9GHGeX5QumN4BI2OpehwkJ0fuPiP4yme2Pm4DfM1iw+x12kIZi+j7ePOuR+zDDKgWG+kgOFQXwRTuJKgzQr9KbFQXEOYBkOPfrwZIucq.sb24JgoBfR8Gr7Ko+I21U.aRN.Loscawd+dd+eN82lC.SbBac2YmRGkktzk3JwEPyjy8.xmcRBsnPkPTuJ00DIKGz9LYtXc.y+m5W.29c1Xl2bov4Yshalmjt9F2vQItXVyX3AAuPvyvJEWRDPGzUC70cBuZ9rI4bTOAgEltThR.v9n8CzuP4lQuf2PjR3kdCeXrZQnGcORrA5JqmcUfqSVQlcjYp0aQjI5F6yCn0W2Tn6qGu1+ZJIIXMEZA7GjBB2m8VJJzJM.u32lG6PuOKk6+KEcMQ+cAkVprjwG1fr1No1v2c+marzue2senOTqPU9IzNz3mBDHU7adVgpNBEcBOwv+HOPcsqK4OCbsuQ+YA+S0+Lksa6v+v+vWCG1gdXazOVR8S43NtiCiebi2ZiVFI.LmUX9mMtn3c6I7D86ZI85Pmwu9D+KDH7SQO8d4JnYo.fMNtCMGAdKsvS1uYUuBXtMnwpENDvx+otfmpz70MeU.c6OUHaOpySDPFZehTki+SADnCNPkef.EseosSxzAYheh9DSR2viMkCCnsYhaCFYc0AanhdwVuMacm8ScUV5RVp11EgzjXOR7hxk.nQ2Wyyzyne+Tmxz5ZDppjTTfSdaXC7KE8oD4POxec83pxK3J+TCxdyWf+jh+9BdoM7ffuTvH4Mo.AUUUM+mzLnJAODa.wYkz3gd9C2g3YqqGV5Do3ta30YHXv0K3YEl4+Vd+Vzep62aiAeHJBhKyXfS0Ke5Bp8e5r7rnHzwLMTCTV1xa3Pg7WXlhDBHtbgZDUkICyBHH+QzBAucsYPSoMWtnE9CoEuvveR9i+MCua3lvwR30wa8.9.u.tNqxYw2h9SN31wbq9dL8pyrxG.4xufUjxsuVwgglV+WXlVRwdQWa+V3ef...vA8A+f3e7q8OhcZm1Ir4pLtwMN7E9BmPT2ZPehx+RQ5uG7+jO.JBO1vvym7n5wKNfGU.fjuCkHcTURidccGmDMjo+sq1mwcu+uD9.9KJ7vgW5B9n.nEAS8cyzXkZftboyfpKDV6BtZImEIAHq+qYBZn4yRuNo2N56EFmDhfKv.8YKaoKsiZo6x1NwsARU06P+cE.vAevG0LlvD1p9tRFbvAIjJf4s+QxTk06WkHJ.OB.5V.xYDQS7IxCIdxzdEkx4VfFtGdMex.oyBjMHqMRtALkIflobiUkbMo3eT8r4nhoHs.djn1pcyWWWmETTCAIxwzrBKUvHkBCNaPeMQJckEN39AhgQqjxAjR8FdWiswez5LzGPyrHBea9aa2CK3+ph7.7Fmtk6TIs4MiB9ZpwvjJqgTQcjoqpR4OgpKAtbXIYnFkqnbhIWAr7cO7Gzfuh6aT1OgecCuJe2H+jnJPHb0AWHYyH7g0jFQZkvLvujQ9l7O8Xg4lhMNIPjImrB4gPbzXirrVok7OyAshjjS6yHYsV5j31mbjSKiebiC+0+0+M3ycTeNL5QOZr4tr2689fC9C8gZvFZ7VF67HsoTAIijxO2de0.EOlggOswgWL8Mvl4b3KQZteQm.ccsmHflLd.g849mJ5ycVsq+qM9iB3wFAd6mF75DCM8bI8aDCJhB.Y3MaQHQOyvepXIRLZ5+BeQEsD4x06FfynO2o7DS.Z+CH6w8QY7iabXKF6X2Y82U..iaRi5yrsaBI.3xW9xrqaOihnh.gTTVbyFDnh8nogn7C.nlZk6HaFOSsIqIqi1W2BflfPJhSgYQvJhMuCKC8XgY9tLDnUndOSSRI7c3FDqUFRNA.SvW4XU1QEr81uBEsuPJ5IAMQ+sAO29tR0MN7k8+c.O5Bdw.Hx+IKoV6KEvmJ3+LAHA5RL9Xw6JbXS89BiVyvZxfVnLEGOyOULiWr7OWAtQMuo31rqwO9L.TGMZ29MOJ78PWgWT5mc7gfyAITWMNE5nMCu19bEvv6XWxvEalYvUcZfmYzgDTjLjKJMo5s4wOs3ejQN3DPrOIlGA5SDpsb4h3HTC+I428ae2O7O+O+ufYLiYfWOKelOyQhorcaWQ6qHmDS7S.yQFMpFcI+yiCpTgLYSG9jHXDzYaupWx+lkz7ebFwaubloa8eVeMKo.B933+nLMo+cCBOZAeUA75jOqnZ.TcouG0EXOxZek0Ezu5xeUUUwOJPc2J4+UrIh0bcf3fHMeOJV0pVUG3a2kseJSZq0qq..FEF0ArMSbh8cEL3xFjHn7vCWpDbhJzkmQBbCE0V3PjfyPJbw5wMtycn70g1Jm8+UUUzVwnodqArA7VaY5tyMn8GZagUz9rxsdCeF25.d2JCJfuFPZbdw9dEPbv5DrcLglRIo78pxMrxmcYP26.sawweWIGqHnS3IE0ulfGuZfWL7laeEFyoC0fAGgAZ1SYvgYbHeoHJdkBsiqZwwEVeVv3goSSr51DeUEsTapNGZF4Y3o5waegfIvABva0Ii.hWu1CbFnq3uDdpU7g5Nd4NUHlRKUQaTGm6TgQ+Ji2BydSspJ75PGoAOCtakQG347zVpozAlI28pVsuP3IR.oDF8nGMN1i63vIcRmTeu89dsTF8nGMNoS5j77JHifL8mXYb84p7IwGXUwF7IkWuggu76YuGkQ+ySdcpQpNglIsnStqL5u9.PDD+ZjUbm90wmgwIZgHfXTfEvFJU4nMDCP0alR9VCutttXI.TfzqRs4eksObUVk7uX0lJtgmno9YNPt8MaEIa9NoTBCN3xP+V15sYB1VArB.PDrmSba15dCtII69C..f.PRDEDUQQYYCtTDsdoCpSTml3L4hBmq.sWqE+ZK7GHKLHB0olGXKhwzaUJ9F.HheFCnI+fDaRSYaFArmEmQCkU5QJqGvmw1Nf2joD9OjzC4IsHvxG.a61.9DNj29a5fHwFT6YcqutaFmK3cOojoK3SaD3kNfOecy.kN3ejwZi0nzelwEVO7tfGE2mf2je.kzbtzk0l9cDa7lVyVWgUu.lCKJekP.yvhwR3124o4WIRPJ71y82UY.Fdp3FwGTsl5LQ3wI1rTT302ANYUBug+hJc57IwAfZeVefvci98oFi6ATmdbi.tN.04ZsI8ltC7mkmH4P2oNw9aY6Cv4DPy6LsoMc7O8O8Oi2+668i2HK67T2Y7o9jeJRmQvUvvXFV92oyh2igW5.9Ta3KOVf06o5jTXqD+u5xCniQ79EEOXb1GuEZeabBbYFR90dF0+yUfG0JWzL3JuHVx+gL8Aj533i24ht9Gg3ewwcZKoOWcrHwOuH5WgVJ6DtGswb6ofSC.DYSaY.1pwOAqwp..1xwuEa+XFyX5KfWeccyGAjrBJe8JbO1YGZ3Pv0h.QdK5gFixhPGFCpBJgf17hmBcj5oTO71Pqa8fEpwKOuhCyPWoIxSQUgUhfW5I7RDdYi.u2.J3jipU.hPgnJYqsVkZ3.TxMlx7wjy+rFH2vbufulnt6yRterUxTtwfWJfuzi17uU3Ep+2TxmRTcBSgsXsuyGY9m86Vyzwg2ykglm5xOj26YXBgKT76EI.ZfHuzDjLgVAT2GJWSdfT6YJSx3J7YxWqD.Q5f+QWvvCNJAJ+u.d3Nq4yno.9L+y8oJBuK.yQDPrmkRN5iv3WeHP67jwqas6yRF0x1W6+M9rq.gk9L4OSzjxAHF+IYnJQvm3i+IvW4q7UvD2DhT5lyxgcXGFl9z2MarjMVIDoidO9y0g1A7fguv.YFl32AfF9eMAesy97CCnlFBMeISIc35eIqYpSgg9O3HUD+8nHgTWvmkIDpMnwPl7eFW0kN09lGT6SPrrXm7eBIWCwmodxPIvNLn1frDcOuL.lbenOS6iXdj0UjYK73RACtr9OB.a0VMdb.G7mYWAx1B2lItsiueAdnUsJr90udyPlwKZa6MivkC4x2O7K+D4S8TR6f81QhJdXkrc07zW0I16KyQKl.JqAhw6Xe1fdOgm.UeORXLh+jk9D5ro0uhe0l.tPqsFLGY3uuA.98b4dUYnD3VVFWyHM09z3lfwTc1jA3QufO0Fdk7CvKcCOaLPTdnXHGiO7rLLbJUpvmhTjMfSLaEwd.wzXwQ4v2UD9qYXtypIbfeuhYwSubfOxBylAWG9.3FOf4kNSPUvXQZvZeB2EOBB1roJfW4+UdGHAuxa45Rb3gGAlXTbTbMNC7R3s6kw+FzWeBw2r+DcPikrhNN.GuX72p3DlxTlL96+6++FGxgbHaV1deuZKUUU3DOwSDiYLiArC2rbOneGlQIO1PElX3CxeExypAKn5Zx7uT7bz2OQ.0qyxHlAaVVAtLX43uxwN42wg20Lvx6aT3kNfWxKWZlGnzS6CAHlURDp1RJKs.G3ISX10xfTiDPEK2GZkllfveSGBOVU4.Bvy+7OeOw4xxD1pwiIOwwdn.YZdBie788Y.vJd9m2lgo68ihLl6IfXGcVTB2VmHQWmD0OMdsixryj5Pg5sW9cS96XEJ7+HWiRVdTW+eeloIG7jh9IpW0qauSV72Ezr+.wSXzGpWrbiXnl0TZ86edjcejqMuqS1ZVAfvG+Gds4Bsepr8o8+bJwjea3QA7o9.dPCVX3Mmd1XvWz+WBO3Yl2FdqOAHPHw9OR9Q7ZHz+q7OCq7Y3lH7vcLTuGqqiZeh901xaeiyEPaAtbf41hJvHNuxcoQbGGhD.IzHTSQ7Z8dA3IvI4mx2MNBIUH+melMCT2HtIO1C3M0IYvMSJF86iCU+xSAFXa52a+bGoh+lLIve1e5Ag+g+e9ZaV2deuVJa61ts3XNlOuO9KeeNJ.tdq1KylO9YCAObYyB1mdTiqisrCDnTBIQruUI5nQKAuEJxbrBBE4nnP3SelxwLV2np+Gbl+mgOsY.dRZxbDfriXy5Vs+P3eyih4C.aivOd6S1INnx+amLfofXqI+q0A09oDvJVwlhC.S.Xjx6C.Xj..ieq1ptcDoixJVwyamTeVQHhPhJ7Zu8ej.nZEjRtmipW7dh3jUPQ0cVOatN6ASzbBH2AK4bAP6DM0ihAOO6GUwFcA5zmM0XVA7lMjTI7Eb.wkgrYzB1I.2eWNqU0RRsFI7YnPOZexvGOB26+Dl7ckHEZCDscesBu1W2K3I7WT3Sk7Oh+KZs6FFivSrOq4o2gfO3DHg2vA2f2Ywd6zbOJQzL9fphrnOIL6SmAjr2wXHD9K93OR1If+rA9JydWixahWYi8zg2L7Z2GM129BpYsuymZLV6gEszgEp1xFlRw9tDUmf4inU+uOY.Z1QIc1WM7VisoxJoXEv3+3F23wwdrGK1q8Zuva0JG3Adf3AenGDycty0uYA9Cf7ABF4RkHMguWb4ttgGQ3McSBpx+tFBDhElDAohjUtNovWk4+MmpoV8QsUXx.l7iOlFI0A2r8.eV.t9C6akd+.OS696xeQ.kjeh8w4BPyXLdYtZtnN29h8krKYxz5DEpTmQRQc3BS+TKEOWcnDqUY9hON941Dh.v3mv3QEp1SfFG.Fw32xw12.+7O+y4FKoA6cLAFTJmAfPGsdjKVgFOHS50l5QSsYSEmHlehBmmpfoihucNZh.PsqGzTFC3NOjPh5Dy+SvnA29FQ4C75a3YlhyqRHA+SUbkajfd45jm.NMUo57RrxZTlWvWJDpxZZo9xjYXOnPwP+rREs+eyA7XC.O6PCAui+nn8clUzfn2ukcGzMJx8eI.Q6bBND3Zsz9OeKJ1lMq2q8vBWorK9Ey+AqpHzWMZCZFHbNGHRYKQNzjnnCj72giZU3ZhnDVgkY.uoEbkoIRVf31YYR8zpj4Md+WJ.OymXduOy0h1OiWdDP3+Ua+tTvR3O4fv63c9NwQ+4N52Pxv+WskYdbyDOwi+3XkqZUHwi+Rzp2SNOkx+leF.Y3YiAuz7a6KNZF9JH400VG+U9sAn45ZHPyCfPwDU0whb1G3va8eYS.d+OH4OkNUnSjARB97yb4WW9y085gn2NY.Yz1FLk4Y13uX1S3N823LPhqaj+FuTmBNpDXNhwd5d7C4jvJWwJJ4t8rr0a0VAHocB.n5H+TGyGX7iuuSA.rhU7BQisAsUJ9kn6qp0xuE43ft97.5ZWm2ddIm44Du6UQXNhca2uoj+BCVqNXHR3K.HqtQXRgp2lPtjnWPJ5re0COJXiQ3sUXyFLIR1vYklMoHarzgTOBfMSV5.bq4KspzpkeU.eQgTd75K75u6G30nyHExuhI5FDY6T9SbCusZdsFn7dH+KdIBnUaN27dia0UxqOalvVyGc7VMxFWIX2gvF8dt7SWzug+DqLjqApyRhDaeiL803OPWJ2Hi+5qv4wAJfWk6XmybwTu8SkuGKFjH7ur8o9Fp4wnG8aCGym+yiS7Kbhuk13OfdJA9EbCmZeXw3JSyaxcFq78bY2bI0a3KyC.fXd.n0jFy1lCCnDrCCH376P6YcdR61WcPCBICUJ+A3RBYC5EiesHHmGCoNBv3REAuZinz3O31mzSDILd4K0P1KPW+dM7+VcKRfkPUXb7eY6SWupUsJ7JuxqzpV5pLxQNRLosYhSF.nZcawneOa435eA9Uj8zHjEu5fYafmPFwCpVycpNSsILK9LIpBTpRjJjds3d7ochEcTzmIW8XFl+lVCC8SV66JThpzsvdptXVLi+1vi9Fd8kbmJaRguPB+oClxz.ul+ZnqTuyiGLHNCjad1.uElOq82HvitfmVVG1.s7ZEdzA7o1viR3Qe.e6rf1oexXkdOCVD6+UeCT2AoAytoYQENzZwlwIu6YX7u61u.dEADOeHrwe45xL31qJvnkrzaf+oJcKVedaFKzkltLGd24fnAFN+NzgupNgVI4jv0uta.nYGBGWXXBV2Ub0vA1.WBSaWmF9Z+CeM79duuOLbor2689fC9f+vvWyY0QGZ7qZzm6f.B8+trH2+y5ug0AvQ.HI5t..gsFn0mCRWT9KqXJz9HNN0lId9FIpuLKeF5+.4ffI+px5Q8WZ3+cY.coA85pYhhQ364NAvvSZxck3OyKLzy6Cppp7O.dHZSTK7xFF59re363iZjvK7BuPOv31ksZ7ieKZvi0iYLtMAOdW0JWo03wdPIfPMroBhREDyBe1YtrQXfzwjLmKZpRwX5fXL5rBbbIWpUAVw+r+RBJ5uo9KxgfFjICk4oXq7Y.8BdwgWay.7pvW.cxxH9A.jqpLiIT9Ln3e76YsxDUgjn1Ym9iJHhs+FA9TG7On89D750YXBFxXETzLSXiL13kLN6vK8A7EIzT.+MxO3qPj9osXSI7JJnv2V.vTHXfmarx7ewT1JjNDT19Jh4FRkB3aPIIz9bcUBuACAuM9UuNfKJs1FdVgeKk7ls4RmKbEyhn0m19IFbSQcr889OtGzGNIfEe0KJk+FwHFA9XezOF9a9a9aeSa688Zo7Y+reVr8a2127CgcnEPi7Ez60CGqMtW1IxPhYBMgXadtMgDZ7WxY+1jTRvSV4FUV5wAbUK4Cd.n0+YxLZ6qRBllv7+xCfiSxvfOS.kQtp4YQi8hyxZNF1A53v.xwU2A4b6Svyg7O3bOUBm1fsbxWygE2giR+3cEQM00JWY+6.v3F6XGI.PU0HwtLtMgb.3EV0JMELsL1aJDz0Tp.3D28wmBfYxgnPNL6rgSylfVeAIJuTYGjCbxA5giAY72zyoL4rhHG+o02bCBuZPzGbnnEyKLrzk872SuYJ+IoLoFZxOIO.SDe1+V0k7szhNS.E2XbgMJUttRgYOPa4oVN9nJBn9SqeRjVvDL7QgmOwJoLX5.d7ZD9NbbKWU1XH68Lir9rdYGGSI+YrCqN74qUm0r9eUVMluIbET0k7CDxwOWwS.dytVk8rvY2e4tgQEGros63uHsggqKd7WoC9A36wXYVdjM1n8ekzu0uRiyUmBJP+fCoUNRleeRNA.SYxSFe0u5WEejOxatauuWKkQMpQgO7e9Gt4GoTf9aw+Kn+f9qMJ7vfW+5.Jj9e6PUy97.2.uctkTUAqCK4xOA8ePMwCSFw7QfF+wN0YNIme.E6qMB7Q8WPzOtQvNndriq2t1MYpCxIdDrF8E55RdYlEn+zRxPh+y7BCeUmKx3b.+yHPMRXngFB8aYK1xsnB.XjoD1gsbbiquA7EGZHSJpLXmtAFMqnMSCdEHTFeqayMATxI4LrfgCkwhXFWSUb.OqA7jLLQIYnNXP70ar4BRgO4nQBHm8rb1XqvqNKDTOGLhz3z.4XDHdBo.KbBQkW+e6CPy+eb2a9S6VVUYB9rN2u6MmmfbfJSHwHZDhFrCMDnJrin.DBmRoKPxjrTbnq5WZT6pEr9yncHzRMTqH5VYPwJMISm.T.AsjPIkVCkRjAagByDHmt2LyKYdm+Nq9G16m05YsOm2uu2u68lRl8Ih686887te1q0dsWS6gy9n3mxc2JM7lPayMZYGvlouIxuNiGhxwDd5+0oiAUNnxhn6Q5+1V7PvOFvbi30L5eZBuzNhcKe3XJWSQJ+bo+LjqiIQtP+gWgA.bornz+qAFaRvj8S7oNVZGpNczYAIbjACwFezCxl365RAu0m1S0VLnt1tU8idcs18TYetub3NntOI2cbloOwMoNG8OjnQ3qI6SZIO9p91dU3681dC3vGdqe5meF20t6tKd+ef2O9i9C+Ca2nKy2l1enxHIFQ+Q6G9xglCzMWoECPgi8h8UsS581SAfojrPr1Mb4dr9SKzTmH7tYRa5fh251aPRhwQyeAGD1hWMv4LB22cjP+eIfhn+J7jHSmllxj.TYcJQgtzdz9yK1OM5L41A58AvkdYsA8uygscdtW9kscmBf.HHR3P.ISjYc4IyKNe.fz6i94ze+8IsYgii1WaN95DSCaFhGV4a5Q.blZhSS4KJGY62mwHjMIF5aZKWmdKIYgx2y1eOhZTwsfN4z.iB+mNgY6MiBC.eFyXBSlGi32.hc+OfE6AfIk9ZllVOg.n08v28T90j+s6GeunPVk.p.z09Of0wOB2VAeuiNlx3Mh2t3hOBwcPwiB9LhbWyv09b4yU3g8SKNtlfa55JQPieIvNCHa.tqsGDkUclTzGFCnqOeWXYwK2fAThj.Mzd7mvJUvxJzGv2zYPxmCP7Pjj1Y.Z4steChRke.WwUck3sbG24yHe79NHWO3C9f3c9N+0w+z+z+Tp+G1bh7TtVw5Mkeix+TPuJ97IcS6+r3H0USpiXJmrtpAQgLFJyCOpkOuc2NcT+YS36+z1f2Mevu25Wo+eUdQ8eOZeU6+Z8x22.t7hAZs9t3J6l.fg4Hgp1O7jGfD.5w7OxNWyy4ZtF9Nad+tN0oNUemFN3hILnwvnprEF+iY.AvStttCEKWmG1PADmcpmfXjVqvrS8c6OeN5Cdqsm.lDblpPLYxaLvrhqzOFCYvGgBgY.dLFLP+3Qh.RfG0gZheBvqmIBDxjAY1RPlvj0mMfHQ.1kH7uxwzBUFooUvaqi2T71ACeW9qNLVCutN+Z2ZAez8KyJiLBQJ.zZXOwiUvuJ+m8S4wJoG8Sp9Ka+putPmhypPPFK5isHwISIejjaXvq1ZL3cjPfz9M16kyZQcDDIeFIzVLqYPBROwgWm97wdM2beQD.fdhGQPAC8kDdXllBh.nKrLKhOVtf+0MTHsEGCH.7ReYuL7l+9dyOieG9ueW+o+W+Sw8bO2CN24Nmz9ApIBOb4xRWB.LOm1uKjqzAS2CVeDr5Shk00e4Hki8D.SDPj+bPLbydmzAoubwtP0+ZpXMdmAaa2tq+SZQ+fLfLT8GOzenOYGycezRx6HeGwn3mlV9j.D9PjjmgaBORCHOteIwUjw+lmmy6tVhaFmECuz8k1OIC809ZGeA9MccEW9kiuy23s8J14ptxq3R1VPO4SRBTcXnyF.DQJCzrPkzr1inWOCnCIihWSlhNZr9nKaiLOpjTgspI0kRt7rbZhtd+QWQfG9QknCEUISZiEcWO+7AD+H6VFMGbL2mdoIqqDXg.IEiLQJXwZyUpawQIE.dHup+dgc1K71VfW6.0jAH9hyGQ+fAXP2MB6e.mcnzYS5vCRco3E8ut7uhG6MdiehA9HdPlR9coIO19C4mvXhNWHZzfmTJJAMmL5HgUfPeLr2Xhdg72ititinEz2yxw6WGYc1eq1+0lZkAzQFMQ48fC+w9pwVPXWx96fCXZMj+8PlHRPbjCeD7+x+l+M3U7xeEiU9yptdhm3Iv6487dvm4y72WjYh++h9a6Ch7hxntx2B7QYk9H42Gk+.M+OzuihOU08bIzso5Zdq1Cgdx.8k9WCV8cL.rh9WMrsu32ykqP0Nps0oA7yvq6AfgCDn5dfptTDo3mId6Kjep7X8Q9K9.cQ9aAKKwMc7jO4SsRcr90kcYWFtLbUu3ctjCejc1+h2tdpSbJjB2z2IE9pezL4DwpVCTzErlwWMuUGoTZx5WGAR5b1V3ziW4gLDwKkUBBTW5.Y8gk6076pitQFigTWGD7Kt5FIsmBfob8+QaTZ7vRxFfa0uLzAj7J2GCij+hEdq+4wYGH5Rk.rstugf+rr5TsXaIdWvWhxlFJlv3odv9fG5dNYE7XD+xkCHBNG.8JlByjx7hs.+3B4epSy.6KvKUSIQ2dwmBc1VAxQTB.Yj8QC2cotMwN26jW5WhQ.0qChuKSR4ehYt3TMaRAC6dHlB4GmwDfB+eq252.ty2xc9rxc3ud827272fey26uINwINA.DYCz1e69E4mLx3dg6xeI0wv.wR7Vhmy.PAusN84IBHPe1J8wo9W1G.rN091h9yx7CzYLx0lDx5HSxAwDYjwMqyX.4eMQ2xxY3nOP0wfyROPQXfdikFXCIh.KeQ20qa8fFZ4R.TWRKcFuz8F.YfSdxSfs85RuzKE6dnct0ctjK4Ha8Vf8Tm7oPs2wFXN5bYb4.nWrZzi1zVS7YQKKGfD3OGkTVU5rDnW7Xzs82VSjqYtnyihlTjsFB5lSqOChiKX7vwhqFbF7uYH1F4Z6uy8j.nrLW++VEtXyM54TrFiPewnjsp7a.+DU3rg9O49ptu2aHVur19gGmu3YLCMQrL3M6WJaXOoUmU0A.eS.d9g2Fvai3qIGZR8q0CShMmwM.5aw3Wrp9cpm1Y8B95LtUriBGsyhiIGL5p2NPX6348QHqorvAwCAOhf1sQYlNJSxW2eQxjOz+u4ncTBTXs2zZu9W2qGulWyq4Ys6vef1Rtd228ci6699DRPDz5P4z3KAnWF5o+VVEoex8GeSRR8FNw8E7.4xT48oXumrf1+489uIaXGuWqpdEhH4DhWSHsz96UPXyPC.eo9ESnXb1iB4B7TWDNz82.PNK0iuZfEHAMXtEj+M5GOBX2piI.rK7H3ei8We8+Syjg1umefvN4IO4B7a55RuzKA3P9srykdYWps+EmD3TCDGkjgxav+npjYf7EkUpnbZPkDJFctHUaMCv5EC9SE43D.Lp.K63PpGQ1ort4lR+CBdTwqNiWQ7Ue+SjErs9+T6pW+FecGmg3J+cbTyRvedSJ00QGD7kNZdIPrOVtninlH1EU7iI+MhuGLIdj4LwglJCP7yYrDO+Ps+aOva6AdmEq1QuY92xhI5SQAMzVmcIQ3jgz4XH4ep9DILjhwjuj9e5LLB11q.t7GpkpteBFk+oCXSvinN39AHY+PPKI0INMKswtrRb.T1yAtia35uQbm24cha4VtE7r4quvW3eDuq206BG8nGMjQTTU1ODhcabeNhbZmI1OaGdDAcVf26GksR+ZYQW5IBv9rrX8Dw74Zxb0PD0aJ9IjHqc7T4yftwt0kgl5OUqWW7+TRkNhQvyMlxwAbo2wKjO45LQVfTONKP68Af65adv0OsAGoC+.2O.ZcRKvm5.LC.WxkdoXGfm2NG4R15s..N4INQLhrbjBTmwwT+Dep2zV1b587lYwK4l3zWRFweqLH8vxr4rTgbbl.F0jFOfgZ37bV.5J0wHw7jOHOBk7c7LasZlx6EdtQq57XjsZH9RaNaJN.fZibgFr81DcB1k+b54lr9xcHizpDHMX+T3panKNhUTFQaZ1TxLkAo0f4xPM2a7dWeoFLKmF30wGNzDeVgyLj7+3iHiv.gbaDO7Q73hJ9RBAK3+XNChFxX+WNKMi8eY+Q3fdj9Vc5ByMpjkLlDyPGotN6JbThpSNZyqAsyQTNhO6oyXGcZEiPSjIQSRb5WyRH6VD7up+Uea3646464Y8OdeefO3G.ejO7GNd5nB8OfRPQFLOTwB4OiMj1nfiHGa.uuc34FkCHOKRxGMPtD.0EBc1cz14mjOk9uf461+bjtR+ut18i3i0d2GwaQdAwlnizTpfjG7x61fVDrdh.qrD.gsvf7SGodF9JuGsOYhRrtGSdqPGuu+AL4tRx.D1oOw1OC.W9kdovwzyYmK4.XrbxScp5n3FxteLHbc5WQOvc6hiNmu65msTfIdaj0NOa3KqagFxk9hFh3i2a0LQFpXrRRDNPYZyoiOk9qiu8WGdcM6kjlREeoEnYfi15gNIN6Y6XBHezIijCT121H8iVGokgUT7xx3VoGTR7i3sC.90BbkICTnC5AXBLV1zvJ8+R75Bdeaw66Cd7zG9XDVZqu6LNEO01fmESCVNLvttuPslYR5dv2lx+Ff6pGZz0aX6y54MvTW7hy5VUYIOp2qWWYhCDVcwfLav.YPGJBDZMo1UdEWAt8a+NvK9E+hwyludnG5gv65c+tv82e79Jx+gjP0QwV5qzKO0IUefLHO+sRVppbeE7Sc7wQOduDdW+YxrXey4dtI.mYI6Ay0wDDz0SpaHeo4D5jjeU7X6vm1Gdn+j5+UquV6oWD8fMPEsXDmWnOrHmGPcZZwnIYn53iW5xdDwc6OUK5FPj0yAcI.b3W6N6b3sdO.hScxS1y.LGAMMnyGonbTG0PGHT73L.LOOiooo1ThDiHwR8Q9WdPR0CngA5OF3FPmx+LC0Yq83+w.Qw5XqwcQN5ES5hclLhhmkUCxJ7TY4L.JifSY4X1C5YTxyN6Iosk6A.fYq8jMvj.bos1EvM5KISveiestgZjQwi5uk7qytmEkYqwKgM38F2mBI2pihjACSi7PuCOMf21Ddb9iGzwydg2CcK1+o3AqkhMhn+ILPnS4U7HpwkSUIc3jcahsg.OB8GQwYYher2NGoQHEPNaMheDYjdQ.dOswffmk+k9+3KEuo2za5Y8Ode+Ye7+Lbu268hyc1y1tQ3ah86HtO.i4mcNkY+hkWsCiQgJivm0cLJVcldD7puErbm+GO9bZ.+td2TrS7lgz6ktqKwU8Pc0K5OYAoMDLkuS8usGuEuBgYSb18X.UVWuisswk.Hz+6bcPegb4lhFRx.o9+t6l0p1ej2K2WLyxrHD9Fb9dzoQiScpsOAfCe3CiCA+x1YmcNxVC5rm8L0yu3HDepnYgCr0SBfYxnBT95lzrDaucFN1JihVzhxLfxKN8+ZBFbvtE7V92b.oIGuZ.9B8q2RWWzn8aZxDCWcEOMfALSdpSxDT3QaplnBGKWSA0yWwwjY8kBvU3hP4yXaeDue9iebyENNxF923YXV5+GWdfE3kNfsGOVz+sFdI9JnAWxyR1+i55E8GAO1G76E+qZPohHPrjXPzwCEPoaSwq9zy5jAiitN1m1+RjTbTegvC4i3jNaU48P2IVwoXI0lbydEs8X2OWWdAh+HG4v3681dC3k+xe43YyWG+3GG+F+l+F3y7Y9LsaH1Nk.7DfDneg8ibOaE7wrs309OCYYCsKen+SvOMfuE7etuzj.vAlF5+aG3abJs6d+GB9mdIcvWgzg4pj7tONaaFR8OVac5uW3U++yd93g2jG1Bek...H.jDQAQE+4F.b8mBfAIXQ+uo6VSPNSFdx83jpE.4qD3gKc580YoKd8bSKldx.6t6L1cdFGZK13qG5PGB1zgtzcNxgOz9VXdclyb1B6sbWLyFuE+9ZWN.LY24y2s5sFGPbrjBKZj9PeFf3KTDF.8jK5OE.5gJQ7ncDtVGBT0+bLRwj7gSSSw2KWwQae5OGnPYjOkDP7ZfB9ElIMYq3w6wr7nI1ZKAvrwi33NbwYP3TkYqJNHX+mCQ9uJdbvvqMphbExoimWj+Xr++hMdZ9UNjYLFoS5+6hqnqOaWPZakyAHrh9SveYv7HLphGqS+.uwT.R9u0zjY3QsehjdjDBFvGNaYCvSYgKbblLBCF6Tb0ZY1vdNvy1ISZno50z+b19jXUdH+CVIjCj7i3u0Wvsh63NdKOq+w66S8o9aw68252Bm3odpLPEGwHP12y.7ztGomQhofumr1R7HqajIi4P6KoGcH1db1gZ+l9lG0YvrvuVuLsJnQGXx9.P7fGLMBkAQ8Go2qd.UjIjVv3U7XKwaAdOlAUObg68.zFlmPYI.X8P8WmJqg9eqTkjoEkaZ+.TeQ0s1L.z9fmsSfnbA8kDsN6YNCNzktcmruW9UbYW5NG9vGjY.3rsffLq7vwS5rHEN0rTpMLds9u6CNzPu9nPObJUBwJW7Q.DHxVMGcZ++ZdC6RwpiZM3lI3IyGSsag8k.Md5bkihQhCkEWwK6R1447EnA6riY.nwLwimilOQnDZhCYFLQwKrhZCxatDO1W7klGMHgprVkeQ.8dE7zA9VSeDuTxMUACIDVv2CpcQEusrAnI23nx+TSMjSxHD8P2l1HKouZ+YfqeaiOxDrxj6yIRxD9zilsGsKO3+PWL9dcc9S9JasdAu3bu+8CYF91ecuN7p+W+peV+i228bu2Ctu669D2GV9L2Kx0xzxG9vVe59ymYeeC3QAeDjWFckNPKMoORef1fn3qOWEuaHlp6IKwmKKf04SjiNV06KAzQn6SFHW90tew3PwRWO9PAdevGBtvjLF3Z+9sCBngGeQHw+jN.MlH+A98xRrATB1GO0ASFr4zeGEJ9PCfdgKK2PH5Lb1ydVboaYB.WxQtrirygOHy.vYOS1fDGz4ZEVmIf0BtG+RO.cDbx0k.HaT8OEUWSdji9Q4E.T1vFwi8mDXjy.PL6EA+68ALlQTyLvx9UZXLN5oHcDKcbxnylIRivxLEFsle6g4aZZpy9Q9pQSpsYZnrgF78kLfSgmMTujtxn5i.JNmIiUvis.O6KoBsYQ8xlZVQBCQlCxsU5K1V4GJJ.qh2H8oji0kUC8j8YC8+PBnewBeh5BDeq8uQ7h9WDRV6.GLGoCoQmSZfZ5Tj5GN5NQoy6M5bK62JGOrQWH+ftlyDOcDiHnFfgm6y84h63NdKOq+w66K9E+h387dd23nG6X.H060DcKA2MNs4gWEni3+BEe4yH8Kz9boqFLxbYM+AJqQd6nTuhOcIKADsn5p9BiDRQ1+KIlDAEsA8Gwm81gGqiWJ6D5i9GXXS.p6Af59AnJ+sE1GnK27tsw7vlLrNS.8rVh.XLzQ5yfwsHjyb1yfs85ROxg24fsG.JKAf3OOZXsj.ZYnnAvY4youBHCPSm0syneqWusFNctMYVVgo+gkWcAY6oLnUXddUm6eAR2hGwtxtx+oipXs0yHhEAP3DKVB.jSUOkE6oC4F9496kxldQS8gFQkMZVOHvbzN7lBMM3C4WRJEeIHhzAVcXrN9pTL5bKFTi0ULCLJD8SgAosBdrU3Kxb34yN+XxIqp+XaA9Jg2N7MgcI3dGXIYkTyCLfoIyvv53GouKmUQhtXGxhfygPCoSltFgN.MNZMCp7yB+S7RDspG4zgVwqORGz6C9W4q7Uhu6uqm8+388G8g9ivG9C+gAWC7Eab0dfAfQ+p5L7jWL.eA+Jkaqv600gNlV4MjbA+9jDEerurQOYztS0iCXMvkFPtTADO04TkjQ8GLnLp3UBtA7wzwibOUomTeCRtZxpJ8Ecdy3gATx9QerYfSj03q28jRRx+l1WpKQWm5tOrL8680gOxgOzNG5Pa+TowWDPlwomty.g8r3RVFUBubfzaT+r5uMU8soEpEju6SRFs3DC73U5D5nqvqSSS0MFWuryvau6A5UPHD8Mw+iA9x1TLpMH7LwGQ4qw8CGypPIZyYeQ9D.jN5auCHFj+ii3S4EFPUBNiUwisBuTvn8CKCXuT9I0aDXLC7Q5uY7Twuhm5C0bnHdKpKFJMqqFlIjKgSZCozmYsOhmIisFdTwG5uzvMiKyh4g92n9ihOCbGi9Wd8p2MYFzejYkKjWdXOo1OpsiNqZb8cAP9JnF8YLHF8QGWztrfwhfEdMvgNhnsA+UeUWM9N+N9tdVcv+G9geX7te2uK7.e4G.9rH+JgiUaFuHyxYHBYmtFvVrqRa7ye7do+qsLqL4D8vxICd5kAj3yd7X.xYkZddFygM03zkizVAnnikAZE+G8lQLqQhLqo9rA7vzl+B7Sh9uY4wR8zpaDP5RL8Bsf9ybYJsd+NSrnwayvaOIBSK6+.kCPShHrzi11n9ytma6S.XZmIa5frVZ9r7XGDiHoTBDaHhgf+n2vTm77vtYBRLQGwHSRGkHpqlLtFXbYKqInzM1g04o30mKRGtZP.rB+GcwTILbboA7I7LCsvPa.+ZWwu6ywo.HcUxMTSZj0UBh5J4SMPyXfZ1lCNh3k.+IutDeoczqqn0HAkKFppbqf21B7VAev+j9kVuIBfTWILnkj7zryWhWtyp3oist9SO.bHA8rrrBT8OHIinceZOvlw2kM9VfW9cubuZhEMrlTVQtGhzJdl3ez+YEhk13kDMZskZhchsG04ck9Fdhieb7A9fe.7r0qO9G+iiepe5+OwC7k+xU8unOK+ab08aj9DjooWRFuNfips11i2Vfu1+wj9ZEXxxMtl5+gIGXtG6WIR0Yuug55C9nr94ZCXLtwfOAZakC7anBhf2U+HE71Z3s.+LaqnOC0y9JA+U6OqbaUVE1ZBc03O7sU6jVKx.sXaMsIR+GlZyI7vta7TEb40grCYSa6qB3Vkua8FhCwjkM5GLKzJW7b..PvSeHR1Mh+Xjg9a2qHXzqnSKOvJ.zG8PYZxkNjTQP27E5uAIBI6k61XHue7ccTyphB+niA7FP+z.r8Hp1UTLDalO1F3ZHwGAPg7pjSRHAQxHQgoigg.pqgegSJQxvcnLKG+9FZ8Gb7lLeJhyyHHkLRlR6jxUI7H6y5M+A7obYM7QgrZEjiky5xeY9FTgPnSUa.4T2R9b+vi8GuqXrUZB8.xE6G0tn5HZrCLjng7SmlSOpivbY.sIUT0+gR+1+9q+q+qyGStmkbc7ieb7e9+7uJdeuu6Fm6bM+liRUJa1j9awq1JFPQPNcPFx2qdEGq.IIuMfmzHQlN1m.xyKej2CnMZ09iXe4komh2hFfP.pOSMxvw3.OH0WXF1KT7c8ei3wH9rP7LiwpBOsQ.5nfvT7I+rRrCRUic09FVhgMb0qRi0oX+GWqroE2X04lsiMshl0FAT2vAjG3HI3Te.HImTB7kNh0WRO.VbBSkSCVSHSEENsLZGNEhkLA6anB9HbDSJJUDCG7FzL.oimVVa1PFV8eC4z4lOAAn2wyLRhtmUwuHpnDOfuLffYwZ9G36VRbC2vmTg3MdX2vUydW4Acc3owUYoM1R7Q4jd1.ezqI3C4+Vh25FeE5GiYDwrpXwOkIHD7eFLplpWmpaDO0kU7r+Sh6CGCUP+iCxToBnpQN8m6CdaevCHzJTfRcZSuOJ5ob5WKKSAaWFJOpjbm8Gy5G.3qREcSNgAo7B7XKwKyPA69cG32424dwK3E7BvUdkWIdl90m5+1mB20ccW3Dm3ohY7v8ZWATeopMF5pFgdZ+utJ8R6LUptV3l.OsIB79VgGB80DTxidbW7w588Z0PxinOfr3IcxCcbYrdnqTlAesTunn+oA.JBVpjWo+B7c8+Mg2E7wz+OrI.KjOXqHktjVp8Cu0bTCQxSD+Z6yBlPMoSah38Jo79Slgu8I.XGZBSGxr8uj8qcm2sNnVsySkukBDsDT.Ow0EZXMZDXLOVKBFVGUImdjxE2DffiLNWGlYotYsQmdQ+imTlDnYrsj9T6UCwTZ1qfuZcp7xX81MnnC59tqMmAfdfz9uqZhTFVVm1vQ.3M.Wi1flE7xnHLKSLSlQA82W1QfyO7dEOMJioENLc.Jru3fi12te9f2Gv2c1ZY8NVAVRlBMJ8+GT7BCj5+J3z9XTARaSdoAH3E9oEyUbDE0LKPZ+waD1OAeqTEWP3SFIC7cxSdRbu268fmIec5SeZ7e4252B+5+Z+Z3Dm3o5cu81qNcz.c6Oj2wq5u4rP4E6G0Mq96Ks9vEO7f5uoBr9T..zm0ytNFiW1RnoQw1R.Lm0Gq5Z1FopePNY41TInkxHCCUl3S67Au47e812jMLC.YalRT5mtJ+R4Utb.8eiAaj8.vZ8CQR4TFS6oNbc44lfgEu0E2iqcLylrCzd.fJCYCcbiuoSmXQfHJZFPzviMNxPJsbzVEAcWHtI5qWSfi9ukQI27Erd7UwKqOIaClD3Hnufn7E1N4Wvp30qJ4auur6S+.Za.OdDU5k2E.wabqLqkznhewrjjiyNQAux9gFaorEa0M1+iRu+EJ9TLliXfeXY+WSgQWe0yO7HwaRYkJPmImEAlKUvEN9XMOoilp3aU6OomaE74HIfkmuD.byV1pSsLYh185TTiWPeWjcqg2WAuWwSZH4o.GN97+C+C3u7u79vyDu9hewuH9o+Y9ov88IuuPwyh+uI.RYB6mSOM79tH.Msrtjf5dX+fA62E3GTf1W7XXczgdz3Z4.yrdRA7Qv1.C8FKI.28pYx3xUmb0EyBQY0YfHriKxut8aDvL+0CDdWlcPFCYwL.nUP0A53xAjBcW5RxY4HNL7hhVkLq4qPkel1X.vtGfD.llV6U12VdE6pW2yOSmapfXHi9BC.DYaouLaRGAVzglili0pWDt0JlYB0lhEdBxwMOGgEBbY3+I+CvHpgHVoO4uBbOLbyoybc7gBeAeaI.ZqhgG+NS9LmYpLoo1LcH7jk7hAlg3n7eXSdExTjNdT9db1CjeGr+WStiyNwEH9nefRzz1pfWm8.Z3VzaVCO1B7Xy8eQ8CY5U09+P.e.vik3i.1QP7P7gT+KqqLpYt99U7lfuSeGIME8xHf9p1YgABxYuiaNIp2sBda.u6AeS7lhO5uaL0G7C9Awi9nOJdlx0t6tK9fevOH9k9k9EwwN5wP3JOzeWJyY2wB8hPXk30ANogFU7Kr0D6uE3i.sY0uW3ScagOg0GPBpyJo2RnLeqj1OhcSBkBNsAThgZo+eSnOFvWz+xFiEezKhw8Ee+Vlk6C.2P9b5Ob4B+GIs4g3CRC.ZbhQ52Nrg73NKni5mXT9wRH2NSne+ubXXxOPYLzxWHyRE4nEzDQJVsICU102s6L7YoyFryumIM+OQovVTGH6rlrXT+7oLHNXfDEYx7oAfxNj+krwsxsy+MvMgwmd+bZMFI+ZreDFJ3Aqdt+Cja7lbzk4Ur6+YaTRNovmDuOfe32Se2hF2PSRG4w4O9pwKkfZvkxHbLU8KaY6IdBZOwmArh9OHIGjZm0xw+p1IqhOzrRGHi3KxIYuJnkqXik+PYdWFvu9U0dsLaAA5Q7h8wfDXqvaqfObZl6ECpGetycN7ae22E1c2gMj7WGtd3G9gwuvuv+I7Q9i+vQ.1p8GR8dKM+0jxU6iMgmkaM6mA2GU2WXKvuzkTnGF7CsQ58UsAU06K6+sGYHlZalR2jz349xxWnnSl0Wv.E+SjlzQso+cEIfS8Oac7Z4IcUMQ22vnj8B4SVfsZjsykjujTX4cqiK6Aqf6xYRJd7ck5U42Y3X5fro92cWeZ2M6MXw0zT1qvolHZLNUPZlBgiQ4Rms.VWbD55yoYcjZRxiNRkBlUqW6z3z0vCwAhmmg+wr.votpjonEDKLaLx2BiHs+bDdsfRI+aBdpem3YaQ+Z7D.vmNBXwrAL24A26YnJJySQibX55hQS1GmoKNeD71lvmcbnMRsLGu7YFtWVJG7gLU2J7X.uz6yYCnKDKNUk9+bjtY+mo3wJ38s.OF2UxT+Ag9Cg0je55t2w2k+Kwy9Ca+wG8gZ+mK7csr7JaRIdAdYjdTYsLqQrsgTGxfXq2cxdAgej9pCN5aPzg9xe4uB9XerOF9540e9e9eN9Y+Y+Yv8+.2OYxtb.gsD6BS+Goehv9ieeE7wkZ6D8eh+OI4gMY+sm3ghWBvzJXH+cud.4fN9bDy88ojmKAfC9d.nsQ.KCdijIcJD7ZFayJ9doqgP8AiyzgLi.cJMp+fT7Ws6sdxMNhW7Q.qeP.ESC+J1+E029ORVf64MRe9BGxYexXeWzzS6hHY.W3ezRl3.so9mmwN919HH.fI6Phy2PJqUY4dwRBvKwwMeJ.3y.4rI8You.XlWvE1EztYOZuscHelspg7MsWtq3ycd6p7u79bmAzoCKMImBbeC3SstR4R7rQ0tSaS1H7rkFdbOAjma+qzO1KSk8CA.hz+2O7xnHX6n7DCHssnohtR6H9A4uJWWfmALDfYVvYLxVYsUj+HNY9R17B.uH+haW36ZEP8WcT9qh2Pjf5dhW4eWGcMup8gMmRC5ohi+w4iS7D2MJIMAh2G6cE85NdNk.gwzdgujjmKUSMwE5ZxPtyroc1e5+0+D7hewuXbq25sNJDdZ853G+33ttq6Be1OKerDs0XeT9ZOvHKsFHduvGkQv6Z+O.C+UrMCc1440w6C3C8OHe1.OX+Mgt7o0Jdg.0KgqUPDnhIa.LG7Yqd8ZQ0FPbSMY0P+oKYW5txJpRM0usEe5WVeCqtKb4LXev1BU4eOZbx8o3q0psV36o9ItYPe4I6ZsqTFz47xSkiF8o+8Cx6HCy8I+.7XCz1ImXgwarNE8LR.VI3eq0DLd8b.n99r2zdQT0SbWDvt53sdwSrJfbuFLSlmwYY1uRGWx+ESy7+cNJlbJdJxifo7k3c42kHPU7yg.XxJqBJeoP0W++zndNRbHUBGSlPX+9sBBGihf7WAu12osCMa0vYhzOggDDHdVnsAulNfTVs7j+C7rtyVcUGoy.EC0E3w536FdLq6fkTKeoBLgIoC9Uw66AdDesyB6GdrA7hiiPuWczZU6W2S5Fs+p9O0kSUkbD9re.C30QxnzOlQwNCWW9frsQdzcfe6692Fm9zmF+y00e2e2eG9o+Y9oZA+sbiXoiBWbtD5MRHBDyDzlvq1Dh.nX6Ei5TD1GD7XE7CIlDyLA.T+GL3etQjkWE4tGq4+rqs7z2BC9yD4hpm8uAuQ+u0VfNB+RYo9mPtsGuWze6u7IKwOpWxr1A5qKwSxkOR0heOzjesjhpmC.qM5e.5FuG6rWG4rsqKUouUuJf407rioCxiMvzgNTYTOTDqNCZSOQM.Yswz.qTUWB.czdpOa1YxAsVq5ZhFbcaliStvFN97xu.ljLA2SC87z.MLhVpv.jUKSAZLhatVWU7Ed0oLQwOEi3e1apnytNZ+Lgln8hLIfV+CYpTwfMpf8Gjddf2p3iN.J+U4mE+sjQZWIYbiHYaCdSviURFg7lx7VVVtOGhXuggtfe7ZAda+wOn+U181cFnjLR+dim9ZcKlMiWzeo809huyUKBvlfpM.Cn8xnh0FwKwGfUTecoZL.4IDRJSP+DeY1bBxm7uIkQyqQ0eIxG6XGCefOv6GOcec5SeZbW+12EdmuyecbxSbRTxrqyaTWOjeQ.9zOP3qbTGTSFVzu0DyCuG6i8ydgG6EdTwG1c8dfPraiqItEmtctwof1h8vDwOw2s4VYUuaWRh.kflL.cvqco6fprNXNSJy4EdKemwvome82E.52xMst9DzT2bfVQ9EmeK.kk.nlzVlneDWM5HhznPNCv1A5sj4t6N66r6taeB.G5P6jY3zeejy7EE2FHdsJtlyVx5xAADOhaAnudM0Pi0ZpJx3YEmYIslAhCCH8HFVea.lauPTyVtqHGI.GN7XvPQ6oWtvAOIhsfBAd8wdh8iU7suvYdhrEUBm6YGLgT9quOt4NysSrzUKcFy.4dhmJegjbAdj3WzyfR.5HkI1nJxUHutY8h7Gi8+6C9HXXIiXo+K3ujufvakygmKV3EcqpMgDLk51REnxeV4QyeC3i2ReQpiB8sUneQ+0yWTQdpql8vVDXGgL2SgPIyUOBv0r8SIV3Keaw6x5+tk3+q9q9qvK4k7RvK8k9xvSGWeouz+c7a9deu3XG8nB4yjcZiri8+doLYO5xeqgOWe5TUxQzU1s+bobQBk.ELE7rubA9V2+Fw20eh+OrI3Za28A0aF7f.R8+P773Ku8+be.vifHTkejQBcQz0+HdOl4GKIQpy3nn+fK.7wa0PjGlbySnrD.4fEPTupsTzqGwNp5uoMR9T.T6+HOnF.4rKTWZD98VMbnc1Aa607ty9Nm8.75C7HG4HgiW1fhSdst2BcjTopV8RcKs1u6Q8lBTI+YwwoFhUaY7kLD5uspPLxv3+ZQ55RwLbVac6yfilfOHMR3I4kP8tEs+LyPoIyhq3iSIqYLOym3BYISVDTZ4qhyPIzxDM3daXw53Kj2Q8lKwik3YF+KZ8rh6xyA4WF31KAQuPwWtFRnqfGLp5PTbkAd5FOndFV1.b.mqau3PYD+9Q+XDn8+S6+7v90VJ+HHI3VUOEHliTwwmVMQX7vgb+6bc7Y.k.nEUa1lWAeu7i3um68dwK3Ebq3ptpqBWrt1c2cwG4i7Qve7G8izlcCabczck8qBvw.R9R7KB72+cQ726JyjhB6MhGXc7v2.dr.uJci9bV9NynzGnMHp4t+mLJdWsn6LZxR74xnac9ryWoaWgnPRJHkGo5Se8zc5bp8CTuNk+6G9PvUvmufiXv4kCPtL61gZYcC.B46YB1osb3WmzYxfMONm4dHCU9Dfa1WgmL.2Mb3Cu8uceO4oOytSm8rmaqAb3ib3dGTN5Bc8C6hvf4WK3N+EzCPWeJ.PYPewrlHNCoRMChuXcSjcA4TGcyfC4IAn6bv6wzzvcJdcJZoBVNR4HQbQas8IwQMCPzqKcP+0HcHM16mRVSSSgS8li5VxDLAT80Fo0UBhmB.QlEjmjjLNIsj8qsFdrs362iNZrT9SIYYSJJAnBxZJdrBdQfsO3IyWzJj9urOaXWPGA5TAvEA714A9UD.T2eI978oQ33T62.c3fvFRG1kWV+QOAI5OLvP3Kh1Lqv+I66BeHNlgDRIrkT6GgyMMs+r+OjDc6+SdxSh64dde3h00C+HOL9E+k9EaA+YeRYzktzmFJfYacLoLAu2wGhYVWav9AlECFAWrw2seEoaNPITs+IN3007mOE.wHmmxMsLwm6gNuKG7ZhmwuQ4Wx.g8eJ7y9jtUQOVuH+2F7X.O8+Ghu3k0yxyAfLA8vVbQh9TucHggNsh2sK8kYXbSFFR9vdvC6EGYB8b+Kv1+QNx1+Vyb2cO8YlNPy.vP1ERrxNyzZT4mwP4qaNq1yne+sgjg3QvnUK4nm4TrjSyYRearugSU9b5xkGZE4pOkcL5ZiFi3YnsTLtyVaQ.DpiRbpvQI8EvLAFCPH3io52ALQV1dg.06zYPNiKwAugNATYvc03VrCjjUvR799fGDeUAP2XeZcAqJ8pMeSRnXMdoUl8EuZbKIDT1eBaT+Q2DTR17h003fv2J7YCXKv6QY2HdQ+xrAoBke8.iYfxv6nDGtKnJv8HPh93jEquskAeyf8h8ip+24iz4Jz3j4OhZyLwKxBEextE5+4+G973SbeeBbgd8W7W7WfeteteN7.OvCjrX2PHz+6JR1fCvziyJskRIQ9cYlYf1mUpqbO.ru388AuXWs3bYYPly.sZ40df15kmsLy8bVNhfU4fW3rZVZ.d9EGtneQdwjDaEmk81d0i7dfGaFO6cbOSjgGi7wIAX4pgpn+qz2xOSaFk8U+2b+pQatwKcNVqyBq28ejZFt6GnY.3zm4rmY5rmc6OPMN7g2A5NNL2UhcVTCHZK2DfoCRzei8MC.uOp1d.ZFiL7LpquJBgJc1uTj0tlllxrR6S6DP9bXlBb0g1H+KsqA5OhO34PGKULJy..FX5hC8oPKQOQmLSdjn75KeCqyCb53Lnxu5nBn7mkZTgy1V7Ci9eM75ndzY.H9MjAafTulTWgo4lvqtKcYAgbAuuNdmz2U7BuDYs6Edoo9yQem8egCAI3vZNNGwWouENEKSoX34n2iTjeU8WJYRRl82DeHZ74LgAP4uE7OihXg7h9TSGNrPrco5B7WK66ExOZaO3e041AG+G7C9Avi7HOBNet9ZesuF9+5W6+a7676bu3bm6raf99.8KV0U8mA8WUOuH+D8xQ+O75.g21L9n8D5eh8qm3U6epKRUQ9zUMWv6w93xsV.e5i1.2e.d+ICfIERlKYfk1+79n1N6en3+P0e8UvWr+53m0oQ2kDZZ34or5zFd15K5e.KoeX9lIgvCmNtLHy7MWanyncIz+C+bJA7P+KcBYlgCe3seF.N4IO0olN24NH6AfKozvVN76LCnzwi9y0FYKiv7j5isuTHJ1cRcVh6neNp3lxKezBSaFu79XtjQUnjLx+U7ko6YQ6q2FiFhDDr29WJyPxCtC3y8YbxI0xMDn6wxYj7eOggNIG248sOleNSLKcVDkqJp2L9jA1Ldrj9E4POi1PZHNoxxJipdM7QOPl30Ec7gglEhzMMhtLnjrC12G7KnemIWv+vRMhh2yjKhQcjUfPeD8645BO3vwSGM4HYR9ObFENdC20EtfUF0CxkYCI+GpeBVO4Q0wc3SYevetydNbW+1+WNvmRfe5O8mF+r+r+L3y+Y+rqPeQVD8sZqePWH6jgkc.j8ieq8wreO98Bdr+3wlwS9I7I.09GU6WKwq1jT9y7Alr7MXmGz2hikby8XvKjqyGS44L3+BiGIfozVUeBQh4fkjzT....B.IQTPT08UoRH9b.Zqh2F7+aJqzSx0P+3o2jWs75EwSIedaUVkIQS5hR7mxyCg5WMZQz+C8EP+XbvFo7G.Xmc1YqeL.2c2cwY28bmd5rma6MVtzK8RGXtthf3PplcdWprxU8b.fJjP9dNcl7mFEOLX8hoNI5zxc0If9nGliTRMnCGpxHC0Z12W7H9d53V42gOpNoA5Fh8SCPGz8+B7bZ+4RavchK5JEQ84dkGzLl6Vhs+Tch4x+WbHTva6MdGaDe9Sh9imSSI+9FjdYyYndxlUW+Qt453G0e2B79df2VCuK38sDO5seUZMR9zIuOHq3LC3dEQzwD51sea1G3KWFs+H998zfetOWrTX0EvEgWsbI8cjkKlYlRSWcl2+tih9Gr1oD3G8i9Gis45zm9z3tu66Fuy206DO0INQRGs8O5aIYmHzeQ9a42GX+8EucPvye2Fvip9PU7m9ChjKVjPCJztfWbrquPxXBALzCe4+vAu.LrL5qQn98S8ejJRi7fnOEkeU7nfOsJYBOU+WLQFyG.uZCHGH1p1uE7q48odJ.BTSna0KWa9QVNA803y620YNyYvNFNwzYN81OC.W1ke4AmDiznyK.CNRyzHw3McjmDfrL7DlJGTaayyka1g31AOvT8VStwGgi77+mmC.j+0LHk.5BsiDZVPeEeiAVfOZ6rM2U7F0KR8QDagVyJGkl.7kCjImA.d+v.B82A2HsDX8ISIHGEWX3yrRQV1xZEZbCzH30Qr+zDdLhOJtmkSDerYx.DYvVAuq3ivVYuTf2Wh2N.380vaBdaA9QG2Y6mZegwkzl6NhIyA0wSVAV++i3Xrphj.51bwGyJHk8B8sg5PbBkKYRhMrWH6FkCcGYsejpoTFo86jFptECLWr4rrL+I+I+I3K8k9RXut9ReouD94+4+4vm7+m+RN.3H4jJ+aB+mlrZbnDOVGO1G7X6w2gl5MiINaU4u5+r3OH9syO7yBddz.6pRVuliyzj9YxRwG3BAf3Lra7U7er.usO3SSj4PNQExz1fENeGqjUPL8+kYAnw.kIPQouVGg4ihucMMMEOBf1FJSt7qpchFqII4kdYW1B7a55Tm5TvmwiOcpybpsFzkGDPGALEjj4GETQKQ7Pg3M1Wrl7LaTAl6hSQq9az4U7YTEdyxdLfOVayBdx+5z6YwuyLvyTtFoehO40MhWkQo8wPco2mXZo.LYne.aTmQClo5hiRRkonhuU6wZ+rGzRmleuiUGIQrt+AwsBdb9hGa.OFViaNRGQGptF37uRym0U2QBC7YgzUBeRiWoiHFE8Fw6C3U5yQgSO+45nq0eTgk6iHnWY1w772h1uq0kJJpxuvFSoiLxcW3e3BuKzOvqzOXsTpyhDIZPGWEKF42AGMUcNuZ+t2k+o8lQ9YC3c2wccW20pmRf6t6t3C8g9P3W4W4WFG8XGKwGB4w9AW5+qwsheefuOv3sk3Y6PS7asuq55p8yB7L.kpNo38CFdFvOeS.lbRFWJW9xbC.N3zi+UvqLEGkdtmJJMfTdVxv0E6GT7eup9W2w8j7K.8G0wnAT0MRcNGp9md4RawhfCI80Cgu0B9Sa3VdD4xAD4UzaibPBWdL.88+5jm5z.6hGc5zm5LqQ4UutrK6xiF.kargwOmBEfh.QTzMf3LVlAlimIznh60fkBh5HtZEbSyZxjzokOtJcBnihrTAKGECCJjy3wFvaaIdcn.Lywf7SHOJfowXeS034H9ULV8KR8mBP0QC1K7PTDsf6q3ImIIlkRuA7wLirA7Fh9zE3Awa6O9hOkp5b3nScTKFik1BxfLKvi0vOHKJIBa48DGEhnMkedVak.9QhCh+sg.FQc4ixO4pz90+hT9hQ7zKCRGODmjvhQwO+cQVJjOcP5BlnBx9hwkyPFxiX+kzqrY6Hw.vi83GC+9+A+9Ewvi7HOB9k+U9kwG8i8QKCrXM7YTLKZ.Q3QUmei3s8Duq00JA0Mh0jvFi1jA8QAOT7vFvK6m.jivlc2aC9xi8mPetA.cTeql1e+rAdVmTb4jJBobIU8p4Gv1up+MFzUhPZZYcorK5+RdHzysVRNSkFfvqXnete6h8my+3Q8Fw55BGcCFtbI.rZaM5+Gi+z91kc.mAfcOj+fSm5TmYqOJ.aSwfIseq5bHrMj0GQMmGbNwoytsgKP3ijAJSaCaQcE0uKc5x07rdNK2csOz4XYEDN0ioocEGdkwOOhGaO9vhxGgOCdT.yiAXNC.LIfn8AU96k2HfggbW.VCBmrfh21.deDOD7xnZz6w5h0SzyHNGC5edgGKwaT9RilbJhEWTnVU6GdKw6W.3iXbrSOcTvjKzfCpiXcjUEymP9v9pj9XDewKZUTFizu+Yl3gduLvl1+KArCZLReqhOIZV1RxUxzbRseosoinRm4hhultWe2A9q+q9qvm9S+oA.vm3S7IvO++oed7.Ov8Gr.0OTQDY.du+m9l9lva6s8iBaZR8js.eg9H0aU4XYFXJseoNE2qTjF5FZxEh8cAOFvq5uKviUvaYW4X+un+v0+mmEI7cSBG2k28ewAPCeFPeuynBSjM1f+kDcWz9kJH0e6s5rBPDGR0eFjKwrLwuYP1G.VNC.CWJIK5+Z+unGq7r9Jcete3+T1b0qPG0NsJ+R8uCVB.mFN7u7Nm5zm5b.3PaCnXJFT9zF9t5t0W9nMTurgOKixVSnvXInyE4d1RcIdMKGGv4akOqVAvh+mFf7FryyPWoXv4llLzlwiE3oidLn2xuxjMU4mdN.vBxrTmkZt72Lir12ko3i2jRcb9fWjacFtNh.hmIhP7RP8hi28BuXHI1YB8I9jgtfwSA8J5eKwinasReQdFxuT+Sm0.WTKB8eQRmpuYRYJ8ktqEzIYBrpilBdekxQhWRXHaAjd5nS7QCUVpngQE+pdcWgHRHPa+tkN7h5URFRGQsCG2y879ve4m79vm6y84SquA9WZvA9ibjif2za7MgW4q7UB.fa6641v6+8+GziitDeg9RxIQ+tz2pK2Vk+8KN38UvK8Mqgu19CAUp8D9djDd59Uy2DfPdIk0nV5OLOoSi9OUIN6PyaNReuI0q6UkR5yU6ot9k3AKvaqfOeDG8928M715KqW01eP8sa+JN.bq81o0y2hhSSq85FdIc3GpKKJPb7YhCZB.mDy9g9R6bhSbhSAfKYa.cUW4UFAkBG0gNyviXwz3jyv1TG.eJ.5KC.c9qIoJd7.f0OhTa+XwYLclnTLxZyhQNO2+bi+CyLvQZP92.JSMVpukYqE6BBGAeqOu4lXfkSKUheLJZhuopwMHoyFYu0M6HjY7n+r8n4vyl6daRCjFreJb00jO1LdZ+2ES7pClNdLhmxeZFKIdjSKdFxQCllKsxxkl4fieS7uXtygosR++R7hLgNIrbp8bjzcTlJMfvoBnSEg9vp5ujEWJ+Smqoppn+2S5XLSBcVJzk3pLhbhOjIjWQdOAOnz2k9nf+q1xbjykegkUqWU90aYm3jmDetO2mOnrlTECFNh+E9M7Mfefu+e.7bdNOmfGd0u5WM9Lel+d7+6+3+39hOnujvKbONAkO+wm9mVhG6OdVNjxbGqH+KIBv89B8g1.nuVf49AvMqexRpCchOkIyAOGAOKlUdoOslWlxmH0MYxH8evJ7pjXpSNhMijAT7L3OSfgy4yZAnoMiN5+3dzlhu5dK1I8+18eWdS.pIiETAU5v6NjLvLbbUW0UufO2z0IO4Iww28o9LSO1i8DO1ta4aDvibIWBN7QNRxvEF0XeXsCFUgW3XaCSsRncqNHibfpNtqfp0EOAmxW9j7DGTRbnPyLfAuZ1BtT8xtRUbDlvGbbEDxVfWsh2Dd9nzz1Jfc9aRLpGuBdswUgcUviM4mZvuncyfyLXJSH5oK7ChiZvgLvSpArGs+hNw4KdMvS6dApdhOotMF3eAO8C4o9WS8M0+FTYC1O3WwAUp1H3qrvnolPesrat86CUBiETsOVAe+2WqF0ocUwOxWoSzxW6pzVJBC36W+eVAoFpZ+VXc.zdSp8c8c+cierezerRvefVB4e+e++.wiZUEu3+YoCjRYVm9K8esNdrN9Mbw.GUaup8WwtU3mt62v9g34I2p6He5i5HsvktEKiIe8.CaBs2DfC1OES15MiDVhAbYQ+eZXrviab+DOp3s0vqRRKNWCVeI.p6UmBdkeB+GBRyJKwf7CX7Rm1+XPmi5+c2nSvvUdUW4h5XSWO0INA9yt268ucx84idh3Yfc+ut5q9pxFhKYuPltavVRhTaTd1nvbdlRaVeWWJMrPKL7o5LcNDpn1Xx.8pd1iMXAmgQ9tGHvwfudtNszcUDeSZX453UGMTjDndujs6znx+k10HdapjezTuem6C.dZ.p6.W9cJ+pSLljLhmotjquZmKCO9TlLXZUviJd67DOCbnd76.zYJg3iQCXqgGC3wVh2Gva6O9BoncPZKrn+uOJ.WZi5LiDidFxuILf20YJiFuTO0Qpsv9yS9ej9wLbvDNh9unk2akBdCgrHBx35H5ADkiPkeDeJ+p6R6bjtRGPX9j1cDOG.hxsnz98R+Ws8Cbi2vMf+C+u+e.u9W2qeiuRUutq65vs+lu8ru0G3Us8CjNPxnVE8+UwW3ea63eo9jtuh8WZ64Auc9fmuDfz2eK78aOPdZ+4d8zLk6A.m5Xl7OssF5M4dWf5AbljnArZ2UwivlZOw6J99oDaGudhqVu51eclSykHGou3SA.xGZ808xnOE.p9PdudpE85lwNC+tweak+ptxs+kg0IN4o..1cZd1+JO4S8TaMvq5J4zLjMvUm52H.aMyFiiPbno5deWWJY4oMzbPzKyTRW2G.DO1FyQBFHRBXJ05FnumYJZJ+izaJX6QU1Ddr29Hu5Raau3ekVVeS.lueHR7be.3k5vhoFbBMiN1BKzO6.5evq+QNCBhR3ogYz5EZqA3xLh2C7F1C73fg22F7Hwa6E9byBsDOCvnxupgJM3i9eHSQdHaRi4UT+hD2p5eYftw9Oe.+F6+j1QHrzKuhW4uj9RErl9Krj+Gw26bJ7+hpvR5ibjNg9ajMVwUpfVreipulXN+Ak84G+e9a6aCu8296.O+m+yek1V85a8a8aEeKeKeKE4up+WiY5k+pEXuvm7+vz6i8x9wiOSakk32e62b1tVGuFTeR9K0QlLdFlj5zyyNV7D.TD.5RuxxjI1S6Xai3wFvqAHI9z9W0CiWJVFiUX4T+O7D.L5AvFneDGYkjA1zdha8YjKseijAn+AmzIa+Gj2Flm5jmD..6.3+2OwSdBfaZ6.1llAG4q7VFfd3yCB3EMTyxWYunu9Q7y8VnIMapfog5iGgJ95SkWkieJNkKDTecqjWYuLCTfty93cfa++zdbtN9wlJDQBBN+LT7PblQ9nSuHaUUwdFycSqIsHrI0uyrwC4HONGtYAKuO6E88F8ohdtN9XuvCEuGUjhuYel30f1E6UdyHiHQliyW75hwj01B9W6.bx+nx+chTa+UG1FH9f7fcfQ+eFsS5D7vQDjjET0gPW1qyfDaUB6i1Frhi5ej+SYUphFcbjPwj1vQKU0ey.JNj9YzGCoao0nx+QVS5F3CQ4nyJFpQG3PSULc5sF80NCk+IKmkwCQu4B8kJ3JupqB24a4NwK4k7RvA45Nt86.eguvW.G+3GGbT0jYzAMXcljOFxT9IY6msqsEuS7153w9gW5iEZl38Mhu42L8g.TWybf7j+S2.fyv.74T7qcCFoUZKyeLzcE6Ts+SWy+MimqEuG52Qx5g9WZKouyXhM.3vd.PGrQ1mo13BmXsXEM4GmUgVCeZZByyyg+mZR.tRh5uEUc094JN.I.bhm5jy..Sma9be1CxL.b0W8Uihks5kJB1nNA8ji6LJCVzFgd6ybz+7v5QWqyxHPXZACF9Kt5yed6MLnWxrKNc.E2rZvFH7eLsmQAxDGRWjaAd8h8pUP.gS347Q+Csc.P730TlA.FrJqj4tPKiM4w2oAsLEKXbHo94E95UN8ssO6J9dcjAavh9uCF9A5GAzhJnx+zwWDPRhRlDcA9xUwwYBsnxGNDbjiNm98yo6dn6OBhN1901Ii0WRbp17K00rfO+euPmkUPh284E5ofIBPc1Q7L4bOKWJqD6GaM7pJlX+Q9ebVbhH8ASGkorjiB++M8x9lv+wex+iG3f+.sca8a8G3sVX.kuzYrH4+bj0QxZanc45mWfWSbenecavK9uxDAp5t5Luo3iWBYh+Gta10wLw4Ca16AO84PFk8WI8RFvJ5OgpXjv3P+uDjey3q1OMycY4lDdICNmwkVcC.pyhPsCT3qzGI0e4rOu3PMZ0K1+08GH7YYVbjOeMW81uI.epSdhyB.LMcV7IOHI.bsW60VeGhG557yxFBL5xpAXTCRdDQ1tkWW2HJaY8K0RNfGu2uOHLmyinR91WJeZUxDGBmXLdPMTA6xj9qAGLqhGaM9fi7pCccO.DFS81ftG.XxMb0jlj.zlH.UxqA3q6CftB0dgGqgWbhqIHXWn3wJ38k3wHdrE3sJ9g.5EsoP+yqA3P5CU2a.rFhMdD0u8jmqiXNskH+6BosfPU7tK64CR+zGT02Tu94HlXYT7kHuPwmiTJr4XUx9BOCnmxKoEP4qXyQmiDez+0gmALIIE5aP5+Pz+kaTpwJnU9ibjif631uC7C+C+ifq3JtBb9d8M9M9Mh+0u5WSs+CzQcp2a.Y.aHseQ+IZ+dceyrNdQNC5yPvW3RwNj3PFPeT9EUqo9uS7wg+i2F3AeKtlapMDseChuHaBEOzT2TsSGkeZeYY.WrrRBBf5eRfRQVnyTEkAkAQBK1aClLfnM8Z0Ioijb5J8+grfNqPaV.1kzbN8Yr1R.j6Afb+tfRYkjaMfq9pulMvwKuN1wdhS..L8G7GbWexu1wO9VC7ZutqS7qSkvzXrvbiAkiQ91EvCmDRzAAP5PL08sPnyN+zAKE4U5wr23lsvLDczj+U60z4DDClbMbFW+Fc8xp3ssBeH3VE+rrG.Z0zTz94yrZUIki9OjeUAnj4YpXobjuM3EMw8EuuA711hGqfGaA99DUq5RDV5Gt3Ch+XjDP3nU0+pIRT5MULHCN1q0tdR3IXEF.kfy4HmUs5JdMWm.CCT6R6mXXRGB8Gwy1PAezmuj9J+E84N4+J9Ho8nB1C5G9ORmekYNfzWB3n9bHqnz+a3E9Bw63c7SFOa+WnWuga66EOum2+Bg+Gn4Pv05rcjBvzionyrW3GRVpfO3k12Lo7wRNw.rCJPE7XDuEOtw.l75+MwyAo3fIV54g.DLXlbXJM56SouUout+DboBxDTRaaZqR2GrtxDoD7CA6Mo+qcL0i0eR0B9zDaPaQ+ujyFHiFG5cnsI0Wu1E8ex+VZukWLAIfq4puFbnCsUGmO3Lm4L33esm3gX6FGzY.HB2lozmJTw8KV68Vk3HF775uFPo9YNplzwS5fNoep3JRnXI.xyg51qqx7YXUeenCJj6UttlpYfOc8xDmbGD7HwGrrXOSS5ooo.SKPWqPyNxWBPNh1TYWq1GIQCtmAQGBp4yx5x4RPJhWpKVF1enNqJ8YaCdrFdWF8gt91qv+Q.9U3+3yofUSbIj+RHFpCGadOO4iHe0vfd4n2CGOhC45.o4NWIGkJa+rNI8QzkYgokINDCGbvauOy6ERxqNsEKx6d4PN6YQRF79RlnllDqnupyXTX+GzDhC2TmWLERC3NOG8AjmMk9Z+pGvC9W1x4I+aIuz4sCcnCguiuyuS71da+nKd79tPt1YmcvOzO3OT33MY+T+O8egPG1h1+n8yVhOzMGvydv8EeS1bvv20Q2C7bC.x8mBSHHdO.ziJG5oh9pKekxHFDOSboyQk3LB9Hdt5+cKvi7IYv634KpmkOt5UcSJizooWmg.T9bJoiGwvMboyZrJKbweZLHIG3Zt1qci0030wO9WCFvC.zS.3q8jO0Veb.ecW2yINojhCakgLSpOI.hSh9UH36KrclE4n.iNAaeNcL1bBwL3x73jK1wImE.7bzONa.VKCXOoUDnz2D8YCt8ecWzfYpmN9j6EeWEXZqGHdL.mrxYA.aR70.bnv0oe6wbDaM8KifVCpKkMkIoQZYD0C8qm+3sTVG3sMy+c2UI+mxvsC+RNf8eg9GSTHABci2ninOB8KNkX1DY.QlWAwrbFLhYKPR7wEGhJdH1YExyDFJAJXRH5lMZ81OvJ36sEaDuWqfxRxX4nhrnNS4+XbfE7pDDhFiat8OrILaL.tgq+5wO9O1O9d938cgbcy27Miu2a61x1evW0.B7Kwl8xVq8Ofm21WAOVAO685F96MdaKvy.MZaJwqOE.AdOrRhQ+Gu126LUn9vNWS5xxLtiDuYRfIu0a+RCnfGJdj5s6Adtjp7QzCv2vaBv00+y7MDZwXB85bFdlfAvFmcgfRRF8rsr1SbmYs3xa60S7DGGyy9mAn228DOwwO61B9Zu1qKds5pqepdUlY.zK7ZWbuDra6s1G5gxcEJQWRRPbn.rn7i0+D.zyB.oBVFPhNOE9WoutvFw5W1O8qJFbDduyeic0BiyDYbeFvcQuyVUWwc8QvIWRf4tQyBY3FHst9gl78mUgWJWAupaJYrGATjhzMmPjy5nhkKXCf88OPkAh9dsU3nVeo9qOPlj2GrrJMG9Ep+k2tWN2KzWXpB94g1ICZWv6juFJHZ5qgeSjx+TPpzWHe2lelsyv.zS4+nkckPI8n8m.+a6U8pvOwOwaeqd79tPtdMulWK9e3E8hRVrK+oLjx8wVC+bb1mTFoNxOKyrTpmIIYuF9PXuI7XIdk+Yes3KOFfgvZ..6Nm9p7N956OGZYuxkk5+FaW9nrxRFVtWjrafuG+nHqiZdg.f3mcdhF1J2777hYmdQEDIYOv+LdnKPTtVkI5doawk1JVQ90ouYsYR3ZO.y.vSb7m.6ZS2GPOAfG8nO5Srsfupq5pvN6bXvrQH2DSYavm80GYb5AhlW+JNvdVlHQorLydIqp5Z.E4.FW5dLvcue14a8GElLqMi0qqhZx+HvqYsy1eEec2pawuIGvQZ6QuQjXZtaeyWngsQFwSYq3QsoSnIXxFArmYtUmYDxazwAPkOy6K613yK73BCusN90jerGuX3E0Gk+Y+OqSP7BuDkQuDE.5rv5eIGnQVuqo9xohT2I9piov0kQ5aC3QAOqAYfNh+MePl1PPmxo7yS6ozrZQhTpcUxa5RKAHR75HKCSyF94v9M3.vkIf8ygNh19Kzmv8Ptmba1+ekW4Uh+8+692i23a7MgCe3CimtuLyva8G3shKoeJAZx+T8mncBjxcj+cU+ekYARwKiJ9hEdgeF4e9HrkG.Y8k7zyo9We5qCeWwF.T7Mpcxc+3556uj9dDrK4YYTwA9zG8FwaL+pDOaWQLhdbioEi5R7Hw3Oi7uU+mMH+LKOP5X8uoyFm1lfTssDgVmtdONzAYosd7m3qgG89+G9Hn2twoNwo9pGjSCvq+5u9foh0Dzk0FgrouoraV3lCJv4nKtEolN3iQj3r9yeWyAKpmHKtVBJlCYZXxjaGoej7lKAk6NxqFbi3s8AuN9oAyu.dN5OlIZKPg0yrt6x0o7uWagbt1NZjO2.azve72wEM7p7QGYrk5KkeO5LiJkzQ+8ExuXpvXfrLffOV+1HdO980bDN96ZSL3GvumzZP70b63B8Qk9P5+xfeh9cAOVU+WmELhgEQk2M025LHTwK7OGAl1eEBfzKccO.j5OdHaxDGhmy+EzuWWhMuVSI95uQ5WzeLfW1K8kgex2wO440i22Ex00ccWGtia+NJ2i5wp8i5kR+807R9LI7bC.xMeLPlHfa0YIub94zeCmZ1TZuTHlm8+YrUPRmIXK6I.82GwSceEuDfVSj28b4fS+n4Lcrmy..Rc1Fu3KKBS7xx2SKwSqExmB.owTn0Z1upDP8kd8W+Mfs85odpmD228ce4l.zr4O6wdrGeqqfa3Ftw3y0cVIvnqTNB3UtI.5uukgzH81yfYlqkkczLiqR1RxT1tB8aG1P9JmFfB+WneTs48nVEuWAON.3sA7qjFTfuIAZ6C.Gbe..nOM.Vg97L5ddQclJREENZyQkLV1sFuu234HiYYh1O+9frDhRcOq3w0weU7dxWKS3bXyPBen+yzhVvaIwT3EGlg7mNUvn7SnujnX3LSz+Bw5J7+.7.SlzsuNdl3vH8E7doni1ppi3j7Yd.K4+ProxTMAGMk8BbcWdW4+lHtWXwmC6KOxQNBt8a+NvOzOzO7Ezi22Ex0K+k+xWbJAFIpjY5U7+U13y72a+P92h7CWX38Uv6aG97TUshmSqM2.fblJaEXB7X.NseTFdI8Eqb.vkDvEEOIgkLybQqR8rxkDHb1Uz4mF7+yAepwHfTWpno7YS70aYrqHlfmxJd.twZbTkWERblMbW7krB8et2v1m.vi+DOQPx1qQ9yhO9wN1is0Uvy85ud.T2oipEutb.M8o0Bz03AcT5psAwUqGwQizgpetPK43.lmXU7j3KBnEiBKEtwNXN9StKpGouirqby3671J3i+w1bfeFv57OskDIX6oAPWWtdfBKeJGnbNvIYmvtkj+6+kSS9dgOltd6BCO1d7dg+UL4zaliVfxRQNveO7Cq8+dTolQ9JCnxOTBN44umvacf5z4m7emNh9fMTu0YWHwmzeYf1RxYhiTs8UBhuW3EcREeg+SA3FoeJekMt3PpQQ85nzFo4.mQQU1Tl8CVN33Vu0aEu8296.uhW9q.e895NeKuEbsWy0TRTJs+qse92EIWCU9j+lNiTi3Cc7dhyi1DQOG80H7GGZwdgeb8+y2CIHdg+j95yy4ddHXo1I0HdiyhUCen2BIwkh.Ja.zuI8UqAJcn6Z+0E.zuo6n+xNJeS0VuT+O9R42H8Aj+Vsei2BfZghKo86tjvgmru6YtOtianOq7ay0wdrGO1yeS..m3DG828nG6XacEb8W+0GGBDp.0kd3PnuRhTpmgOIbB..f.PRDEDUfXJWn69U9Y14Yn6XtuHPTfm4HsgLohovo+FrxrniNdxCJ7mNJET5XLsoJA0pMu5ieV19abXtFVZPwZEsf9lUvy2xVwKJHVMcqD4kHL35dy5f65ek9AeHAKCVqfGI90Z+aI939qI+1D9PLRoSJzz0IikyU75kFbicF7uQ8l0Pp9ZYhI6I+m4gZXj9VvlaZYwD66nbp7M2U1B9UzeqseaQY2D9QGzU76E8CISQ+sv+KvK8qJYM4uqg2R5aSS363636Dus+2daWTe79tPttrK6xwa8s9CB.g+E6uXoqzQXCp2H2eAdrm3AwK+cTOaT+UceuM3CeMZhLC5eQMW7GXzDJYj3ypdvnADVAuAMQRZ+tY7oi0v9a.eaPSZ6wQYSMjEtQ+A9piP9+5eIsI8W7R.ZsKx+VQXEMuvty.thq3xwke4W9lpoxk6y3gdnG4n76S..enOzG59ehCvgAz0eC2Pdb5pFy8oSYTworVcrozkFkQna7f5wUYal8X+eznvE56ZmsdMYwZoy2jdwnj6UdN0M55T1E7FydSXDJ78zHMw6B9Z6ulEnkMJ8VfFlS.V80pIIOORforguQE4oyEeT.yQCS5qrehWc9jzej+Qk+6F3kQpDs+05+QYDL1H9nOcE772C74rDnu8CifCttgirjtI7NOKNy77Qba09OJCYPtdEnijcr8OReW5poirJ9AYLp2K+sbpEq9rGk+hMZn+P4e19Y+2BYhfO+mxWRvjvNrS+Rbok3iYEv8ntB8nE8+dHq4Oci2vMfe7ereL7591ecOs738cgb8hewuX7ZeMu136o7eP2.K8e.OSCUGMNuOK+B7c8u8Due9gel9M85lAD.vmyQ+yIWOdz4MCvl.m4FpiGFz9Fnu3hk9DH9Xvels+360wdge1c40abyO5ZOe9coew1f5kQPYj7pv9hrO27e5lSeAs75xT1JCoOq6jVW+0eiKpiMcc7i+j3rm4T+S766vO73O9SD86620Mdi2XmgFKd8dNb4klSsrsovX7doPicpV+g2OxARxJHNIp1KtVds.CjY60TjI39pBoJnK3+daf7FXmjwe47DuH5fhuErY1EkEsHyNrgonR2mCvy1XIy6P7sT9Wtn3PKmjsdfWT10xDSi39gGUGhp.jzuLJGKkEl7OmYOMH+Q.2JrWU9GtfGjAsjpTdST+vBwWDEaM5WkSwT6o0QMteOoCg9Q6XHIEQmoxNRxJQQqINWJ+hlyZOUAHclq7Z5EDf8EtmmQ77uQBKKc7MzXJIKQfF.9W8u7Uga61ts+YYG9e9d8FdCuA7Y+beV7U+pe0t7g59LXT0u1ZNwzjEB7RP5.kXGMTAKwi8Cup.296DZu7wT74QONaUqgeNuyXyqnOj8ukOW3e9qj+cona.Os+JsNZ+nxqtGD4V5qa8DWKdjlHQcKQKzxhleCs7RnCPRxPFjVT5hrhS0e5+hIDPZ0hGucWG6wdLfY6SosS..7XO1Sb5ssRtga3Fwzzg.kt4nuR2n0rXFl1ldKgM57kBTVlbDQ42yQQ2qSlomlE8ZFBvhcrJyjs8hGJ40nqKqzPYA5+GYXNtNwG.7PvuHHBVEu9hApkgs0NI3rbO.v1GqHcCt3gPTnOG8EwWF8UMAgk3QE+JCIcDuNRn.OcF1+dLSBHEMESrdcnZK5rHDjeL.eLBpr+pLyDgrRW6yrBIs4LAXJsFoOsIpYCTwai7JotnSVneNBNk+oyrLgAU9qxnj9jIGG8d76qgWoewtQ+qv+LyG09k1vfzeT9ozOss4etxq3Jw+q+H+6va7M9FeFcvef1oD3OxO7ORd7rF59o8hNU7rslxaTSPWrcT8d92E3cD1QGL7C6G...Xw6REy83IAf9Z3RRZxmaaqqo.e3ZVMKndHpzO3erA9WzaVi+WEeWpR8XX4rYDIiy.yS1pOA.AVrb+oP7h6SYfOc2IdeFt6CJs3GsbIwVXEPJ4oPj3tom2yCa60QO1Qw4N8o+X76QB.O3C8U+pm6bmaqpjc1Ym9ic.GCLagsO2l4HsQLLZ2wqItoQFls.wubrTskzi7P.kOm3Koy77rrFOl7z.zJdlUkgbsVY.Z1Q0aoAijitZIdr23gfmWRrJo6FwoBnsDutO.hjYr7D5ZdQvz0oel7DD9OCLtN9nPs+v00ZDORa9n+abHwb1BPSugksaCiROqLC.YcjFClE+WIXnN6Ci0P1+ETIaZl97W.Lp+AVFR+ExFYm6ZBdWvK5ubctoIEDcKEdUBT0+WH+cguFwGzHSiLL+jjODMf.uUvuY5Cg9VV4E9u1+I36IA8xdYuT71e6ui+Y+w66B45lu4aFug2vanDLcz6zXP.o4WRrLvK5eE6WQ+g8gKwaaEddQcJ8QmlnzQGOI8+5KysXC.pJ.iDGxfAfU4+EIij6CjB+ai3sUviE3mVA+zzTavnqbF.j1e1hegIbX1vd8IheY8Su0VcOq0xFlIVuHazhxVUC2MIuOJ1uqG8QOF98ej6+2geOR.3bm6re5G4nGccTqb87dd2T3jJbPzcpwQJDICjcCRqSFg3rGqiGyvZvseNxMIqKNBiHAifWjq9lLbdNeW.zxhM4YMCPlEoN8mM5KY.5YhGLqzDexeYpDQ9ige+FbI51veo5L2ijbe.vYB..k2I.gxbGu975lRzbDfgQkOzV8bdTr8AudFiOdd+G3k.1tzQF36xSlngKxePYbTuBuiJOyLEY+Cidl1KIVNyCIFcc7Vq8aE9m57kM0lkXQR9r+m2CU7jeLKaio8CIe1Oki5OHofWpiTCEjybg1huovVM6iw.dCo7yE4WhOpMoMTRlXv1J5VnvJgm1wFvQNxkf27a91wOzO3W+d79tPt91ese63E8hdQE6BJKGOjp.p1IkvBTuOUjJXxO6Q4o8mE38DuKX70wyCTLNZ+4N2wmlJceHwPsQhn9b38KrsRkwMPeWnOVfOjf6CduWlCBduiO1CZqMC.Az9LkLFnuzB7bPq8xvkDUOlDcs+TENB+G9VhtubVL..ddGfY.3werGG3S7INI+dDiX2yctO5i7HO5VWQ2zy6eQ1o10e0SVMSbHu15MtWWLap1m8vY.GcfLHVcVpVezQcg8zDeR.PD3rKkQ5lNelyiXygwpVzLYmbHKDeFPgedbi2P+oPTbvvuQ7sMv3LzGVBumwI2Pi50bmlwZXQ9mzWGAtq7uH4B9eawObINuxfVr8Yk.ZrfEtnXbxxkkHMnEdN9EI4FJ+mqkPe4fTlYJIH5l4+LwkpKhN+4hyAeE7nh2FqBpWHIG39bU4QamC3C3RxskQPLHmVz6IzOwOipXh54Kwqp0NPaYpFa+DOvR4ejffga8Ebq3m3+ieB7JdEe8+w6678xLCeyeyey42a2r8kwf7saVrqBaPYl0h0OJB15CvakO0eT7V3aVyRU4KMgfYl3YLyhIdt96N5yBPmByyNzS.PM+vJih8f9XA9j+2.9LJ7FvaBdj3Ah1VuQu5L.jexRw23kZ+J1e4FO2J08h8A2X62q8Sg+mf+Atoa5lVgQV+5QN5idF86wl.7Q9J+S+lOxi7n+raaE0x5vA5axsYFjPbbWVCXNJkvie93rX8k.XZZpUWHiy1KLIUKyyfNNbuqP1UR27SBPqScpG31LKBPNwofVbt2npoju8QAelrvdgOMF2K7K8lq3mJGylA+Ssk95IMgJ8mN.zm5b0oHbE7cLXk1OLou.g9Z5vh+k2Kppk3IsbklcbQ+e2vZbInTmlzto3CJ3YzCV2qSoPU9WneO3klfq1+mQ9Wz.ZvchGgbh1BlxrHGUttdiwd7XE7ROVIJbk8UdiuCzQFjVSBIvS521Tq5dlX1S7VTWRUkw5Dx2Y.Re.g+a9Cdcu9WOdsulW6y31g+GjqSe5Si22879v8ceehZxpcggpmpSAe9HJWsaJ1BDyfckIAhy8YjhOSHHwCg989rH3X2upy09OSHvmaTZJ5+szuJ.fMAqm73hfkCsIs+24zX1oaEOCnOhu8ah4Sv+pshRqxADWm.wdCaXF.zYlf15kQhqx+xF1qsA.ElZOeG.Dsex+ICltPP5y847bdN3HG4HKpm0t1c2cwC7U9pOrduHAf669tuG5M8V992pJB.3lukaAYVQwjgFNDqel96kfyZBBcARKI.KUbCGhsxVCtSmP7hIIrx9MfA+E9EX3zdJhgzMSbz53hePoeZfMP9XWqG3ssGOaVIIcDmjV1Tl4NwO0LPnZTaWl1LNwZzObTXERlcI0.uo3mxeTrjK3WwIE2A8E7I0pAqIFjzTCnyjSR+.i8+roQ2bZeQurgrfrb0yjKIzpIWj4RrB8svtL0eBGpVwGdQuN6FVR+PjT4+Rv0UvSdFjNrtPxKQtIz7R7AQL7aIuj5uo1FcXuT9AhoIjijCz.MrskaZ0Vi64d8WOty67e6S6u.ed595K9E+B3c+dd23nG8XpJWIO+Vv1zVz8pzkW5LHJwAq1C6ElH3TlnXh2WEu939o5xyLPui1.OlyYBp795vmy17hFMIVpchxmR9met6AnDHdAdwlvwX6uZK.zC1KO8.sC+GrxUtbJzuj2+tNWhtW4+7y8+Vz209ikzhsJ0mglnMPKIja4V1d6ji8XOF18rm6yp2aG8KO5idrcAvg1lJ6lu4aovzduAkLmLpEwIn54SULAPIaeMXm93h4dKvm6nmXPlfjMkcxkvMbaoN0ldJ6PSQFYtwGMijO3mCkH1VrVcOYZ2N3Oj3YbWf9qf3sCuHNE7yQR.Vel.Bwg6xZ8mITwG8tLfTF4vQ4qQP9RBYCsYS5.Y+bz+IIFD2GIFWmUf9esoodv9ZP70wO53qt1oggIXUYQYXp.0MjpXT442z9uHwUk9TsUpaHzuH+DgrKIAvFB6aC5aI88J77wfrWWU5ytPwPq2tnyENqX9p361ebVIJtyR7P5+B6XmV8sYE.JdIwKyEGZtMHazrOb7u7U8pvs8c+8r0in4YhW6t6t3O7O5ODejO7GtEPwpN5couQcPRc9PuRvDyvT+2B427.dfJ91FFpROU2XH1S38za9nX.8XFRcqhe3kHOepilmczN6+mI6kzRxfIzgsQ5iUa+oeVjIKE3kYkCaIdmIg0v2h+zeDGKGDPz9vS6jV.HMkkzOyl5+bDOA.yqrzXAsbNCud2+UlbkW5+bbKO+aYk5X8qG4QdDrKl+X58JI.7U+xe4Gc244a5PawztcjibDbS2zMgG5gdHvtttezL3C6PFeV3iOZC0ZuQFYV05v3n5UGy5Hpoeek9kK9ncfdRFRFsdAauCK5voiKzMxjNeKmt0HSUHc3Q0NhGaDuJZTCZlDPtzTsZJN.NrlSbZOR5OY4rBnNIfPeoyHTlC9ORBohWC7tr2SaORfZlbRYDgi3k.wpiNIICJGCctELPlTQ53gAjps+UgSGI6E8WS9YC3gfGJds+ebDbotBnyoNNyVCOJ3g6Bbk9HyhQvWRspamo5pJdRe57LcWuF+2KKuWHfR5mT2wUdEWEt8a+1eV0N7esqG7geH7dd2ua7.2+82k4YhNgsrDXnFYL6qwJ+Mj0BlxfrT7QBoRfRAeYl9DGPQRwc7beD4hSRMX5jkG5OzNadVnej343kn+Xxz1CcJ0w.dR+A8uKD7l7n.hL4k0NE.YckKqB.hDey+G9RZM2uEm0D8L.XLKrXFFD9OpceYhM27y+ErP5toqG7AeX74O8S8qq2qj.vINyo+TG6nG663Ftgs6bE9le9Oe7POzCAc8uarqHnBesKB2AczHRn7VlRlLx1hqVs6LSLnkoKq1g8a.5Opb8YA.H56xiVWvDJ5FrtkqEFc15clKY+nS1T7TQYU7Xi3KQP6A14MV6jozI+O2ji5i+Wa5sxxtHXgzajNj68ez4sFHrSv3EAR3jwP0gNJIMvxr.uH.z.hk.bL3ixugxuDqbHfdlYcdO1N0obM4eszoCt8h9sumN4rfmD7XM7ZyWZ+TKOZOZOks+3MYJ9KzOseH9fm5zIeDDqxOhmY2TWhfjO49+IBjndgKAXVx+uzW5KEeeeeu4mUtC+0qO9G+iie2eueWb1yblxH6J1bx8o93h.xCXJtLbubuUwqIXTtF8.jAxFwqSkutDMyRMoMmc6Nl3o+mON0.JKvF.R8mh8JerlVAad6zOQHGNOvmmBfCNdm09uboHRFuWe8uuI6m1my6qGAvwOtBe1yjKRpYgfqWtY2OPKU1i7nGEeteueuuhdu5P8O87u2W4q9facE9BJYeHJNhC0xzEW8pUC9GmE.jwLVAQREzKSKHU3xr2e1C1.rH3OPN0KTQlKOtRqQGwI+6kjJ58OAd6hD93xE78avmDflbFR6jGMvohG6T0irycWYjFQ9I.giDS99h9OcTuKRdCq63wxjJB75HfDZkiLR3AReZHFemAwphsBVg2lWg+yYS.U9JjAHlMfx5XGxOo8a01ehGA9BeFIsj5Lpi3rVD8mA5GifPB7ZBR.fwj5BNf3Da0E7exY0QcoJv95s+.KyCPou6XmCuS6w66qiu89tXbc7ieb7q9q9qh698c23LmouAqowonuEIoVlUlRToR+e7uA6uh9eLRarP+QsiI9TapUCZernJD9M36OEf5dBXt2+a8t2IyhAhrj9AQhraB5GDsU508ezJ63rFE9uNOvO2wOMfesW.PQxyR45rO3f13f+J4Qz+moxu9Q.LSxawLUWFrKGPjz+WKJNxNGF23Mt8OA.O3C+PmY7dkY.3y+4+a+0evG50+K7MiuospBuka4VZB1obcKJNjCEEYCLr1H0LK1Qo.cgUWJYgPra3HQsREdqr4+VlpAh0cYZhJ1sBFuHHjQQR7Med8Q2Sesl3HlApMEuuJ9feEFSmJpnx.AmhRqa32VSwoUvaw6BflB9xMuyDc7HAcijxJ7OhQ5FRWMPXLz6g9OUlWD.I+GN9F5+GSjHFnomA8ifsqL5p5T1SxmSGngg8gP1.B4cYeiDzu8Et2V7g9nR6O5Z8A7Trtj9Y6WzM7NuDqKOa+J8GvC1+QmiHOZdcU9w1cJe4uGiFqWWAe1+c1epO1o4zSJqwcz+6Z0W3+a8VuUbm24+1mw7B74785S8e6Sgeq266Em3jmL0UABgPwksNxbVPW8ULjHfXuX.ko9EBFUiPseY8KjewfvR74rnBLbD3J9U3IOJiQxyjDOvO3+T+hv30S5Wun+UvK+daf8d3tJ0+2B7Pren+dEO567+EO5eBdCXblMJ6..M.sv.d2Vyfg4cquDfVlvh3zX32WaCCdy2xsfsY45A.N6YOKdf6+Ke+i2uj.ve+e+e+S9nO5QWK94pWuvW32vhf9i83V2QT735s32Y2ICR2eR.hhv09OcdMtY0TxVBpIWy59..V9NrVD5pRRRc5T0BGj5z0yYdHwiUwaNyTfpi8OQG0iWc1xXvI2if+SAlJ9HiViA.rXc739.fUcFbtH9VLRknMKAX0DWs5WDmNiIZHhEqteCzqkiTAQ.IEOBmACx+zkYzpJ7ubuJ8sUoewPsiutbMx53mrU1uS82nyzGZ+nHyVN8+hN4Z7OPFmVpJw7QBbKLnFbGU5WHukxjPmrPGceLrDeor1Dd8utWOdsu1m8+38cu268fOw8ceYrbrTWgIqNNE+renDnriQmor098Xjuws58oTuRoUjPruA7xfO5tL4LElmE90DWxi+WxaYpkSFvty5BE3CM.j5bh8qpyl5OFhGUkRhx059BAOe17I930.b4xJxONStdUjFxuYQV5vP9H.BXSSXdNe5HV6pDiOjc08wCSF3Vu0W3dTS0qG7gdHbN2+ji2emwa7k+pOzo.vksMU50bsWKt5q8Zvwe7GWhJXH1A8bjQ5zPshCefVYzoom3lAeN8QlD.PV+HwWxVCxH53kb.OLK3G2o9QPEg+yf3XI82W7MDXU7gTieP8GqCLW9ReGhJJh7YwkGXGyhhxjYYR.csSlHCyRtt4Jk9JO4S8EyiK0euCnXUj0e1mWBMK0uCjzf3U51a6d+2jd5RfPeC7Oz9ofU2K5KsuvEGa+od6B7k9MencjzW6KKAri52VRe1OsB+QmpKbDJ5OE772GBDwfCY8mrjt9+Iu12k.TsXDOCdXFt9m6y8+exi22WD+F+FuG7nO5iF8uk.0p8Ap5A7xv+eb2adTaZU08B967JiJChBXApTEBxfTLnHf.5JNdQLpqaZSPSZkDucLqNcRmVSubJ8sWqambycshhceijj68lAyMCpfLnhjDCb6NWiJJJyCESRUEUUxLFn.DJn9d18ebN6892deNue06GfFIGs368848rO6gy9r26y9L7320I0tk1uQ0QmHiHSCm8mGou3AsKw9WX5OHAeA0iVr2+qMI+VdcVAXIvoJ2gucJ5QAhsj4SSTVXGDHaCK.RgH+H9QB9fdjBO5gOH+Ej3aZbBw9p8Uc4MlPRF2vWdy3UepuDD7mCxeJ3hbfEw9ChF4emFGZABTMDAQDrlC9fGIfGVty65dv1VZ5byOuK.f68tuma8QezG4XedOu8XgZ30r5CFW2CdMPiCTOJH1rjHkf907.CTt85XuM6zGGb9n0TwGEbo0JQBXF47eV6Wquoq7ZEVMLBbcY.pONtVqdZQETrVlguQoyCd3vGywZmvpR+SS1myaNPd8+g3ybdFAuN3oEK.QMQ5oa1fgAmpfwM7nNBcxue1fF7zCKY9r864rQD2OI8z67neSir8iY7Gr.oJPF96Yf4R+VPAQ9PqXz46.zannw+IEfgxuf9ZLnxrOiwv6EMnTp45neYYveG7sObRm3IgS+ze1+w66RtjKAW5kdI19IA.w.eiQaEBHvcbLG8WMXQv5CDdnZmcFF0QU7GCHvM9HyW+AI5WDae.HVf.5d8R0TqvqalZ+dKgZ1HiF0qC1OH72pegfuzv+H3YwgoORATvUchjK7R0gtS.P9uZfD81O7dbeo1lToSg78L73+YNFBce0uyZN0LJHM2DqYMqAKZ4tum6AWxE94+R4m2E.vzD9padK20wdDG9KegZ3Uu5Uiq6ZultYDNbs9akQ+lHBdNOmYVTX7tZOLKIzzkKzrh3LMHbkx7V8GmMq32i9fRodZF8JN65vZyB2hzLjwi4v+rhlAOO6y98PgR.pBQQWCp1kBjtN20yrKpudfIYhJwrWcmR6dCn3aFuIE+BFS+gY4CqdQmYkks+2LnvFCUGUS4Ypi.7A4uQGzZ5as0bvegzwBAh0vk3yDtq+Wqm0+UrfLKKG7fb2SzT3UZZIK+c8XmFU5rRS7oBQmwsaySLbo5Tt3mfuw.lAWgnQ8uRf.hyRy5+h3OLSsF76wy640NdeGAd1b4du26A+U+0+09w6CNO61pcq2j8a.nx+nyUSWkj641rGOzr8qMLoa1pQ.WspAX3U8Ip8Ub+GJsLKpSbftTmz8DDeijJheL.s8mjL4sIaB1nk3dnhm8rI+TxmG+0Tq38ESk9Y3QDd33TgWm7mhKa1+CO9eN+a3sI.c6RD9owp.9KaNH9q+WdrI6WgseExtGgGpaD6xNuKXUGvAhEsbm20cuznm2E.vsr8G9Suks78+nKZ..qYMGL8MV0swT1c.PwT7xEcVK7qfwhtaIKBlkfIX3M773LNsfB3xLRoNOOrjyWMsmrwXy1MT3ImZoQ9Q9edvi4BuGEnITLVnaWsCzVeN+I5lojO5c5aqKsl5+jADfvsI67RbEVyY.KGzVQMjvznIbaLFMC+bYTvH7uE68JKe+mGCiEjTePhI9WZgIv7mopL.dNPuha3gkQkTUGXiDp2ZKrkRDdw3+A3m4dF2lvpX+V+Pwn.LKiM0.wo+4A8q3HOx+Eww66a9M+l1w6C.A4OJEDtICyEU+EzXPyIWTV6lHJ9RsM0lNDiGyylqPGGwRne.+Dn+vOTa2ZFQqS9ZoVKnOS+rp+HndW.XsrEnNFqXWfcoQkmTmWuBR6RvPvD9xVITUIXBetG9Yvk+0KAtVMlyM.n2753ek2kXCy3RehIhKX1r1x9LmqA3PlmSBkpJia+ot9+GzBuA.2912N17F27cN525C.3htn67dNoScn4gQkUulUGVmdPQxvavJ0YnnFz3.EB00SWhd2+qYU2iBKsOCfO6YMBK24ehUZuU.mMalsDCN9oAYlCZoQ+s+aSwOu9rE2yvJC9LsZZbN4TkQ..0MPxjznewWmVMK..hE7Dub.hztaE3Yf.2Qsm5XZ8rJL6H1NCuzjQc6Sf7raRF871.HNqWJK.T84mIMgf6GadzuGfUvabqUzYZnCj5m88f09WMZX8IyA9FgNB+ZvDZ2q+MkWaxXk8M7O.dR9iL9C0mBoIPWdWhK+HwOIqctPPLeoorFzZfcZm2Y7Nd6uCbBmvIfmMW15V2JN2y8bv5V25.mYJdLdHyTzr+q0h5az53sfI2A7.sxyHmGSYPysCK+UsChVxii83Fj.8ay9GBsu4n09ufvl+iweMNEw1.2p9cebQhI5.bbDnQV90ZH2lOaGHCur.vW+O5Q.WH3CYAHTb6chNxQsQ.tugvO0Db+GmA.tOhKVfEZeRn+yEdSPRS7d4K28ce23I29ieYi9st....1xl+9OJ.VnMAvy64sG3.W0Af69tuqpgU3y3mSkgNakQ6C.1HltI.AfeV8UkvVcM+zJ3DtzzqxCh3xrY9EZgdb.mw0s.iV8.L.Q+J58TYU5fGN7V.BNuRZRQk4B86j+iH7yRJNtim58A.cg.AXA3zMCDB2rQ7Bw.AGypi6TWm1DSl7GNwxNnLpcfybEWMXM4WxITG8SOyo+Q3mzUHG0FGvv6HJ3XiCpak.eXyExBfjdsRi7Z9ICneCNSGH4DlB6LF7PV9q+GNclwbJPId13.ei8w3uB+A8ROHbFmwYfW3K7EhmMWt9q+5w4dtmCdzG8QsNGWtLZcfccbu2qDzeCFxR5ul7mz8KBkk.db.G3AMlt6YTaxy1rZdIMJtTrKEGUebxzopHVuG68a8uJ8LUIJrTa2sa7uKHTC2pphGDJSSA8OI3f1jeJLrkWe....H.jDQAQ0kcD705DguRT0rYTgmmfTuyeX7u3MfKO7Ppb6eNY1ra2b9qFmmH5oy4u1VEm9m7.yq1kazsTvgbnGZG8Nuxl2xchGZos8eYzuML.f6+G7.23C9POzI87268dgPvgbHGJt669tac1BD8JRr3ACnZB40XuxstiZ8r5WKjJkDcDDm0O84hlogvXAqnczSy7iCnAuFkp1WGHQS6qhc0YoMqMF9hCuS9UNRLp2MUyxC1auLB9ph2jT72LfPM3Tg2NC2khsNZ9xCzx.ByeDNqYpQQF41PiHUCDfbt47euSa1N.ILqscIK+X5Wsa3sqNFP0jBqCeQMsTB2feA7SdtEU2TelgGns53fQJdebgay1WD5EvSP+QMVn5lTPEhvAlPlmyvSFh78Q.gKqOx5ThACnCJDwNaxgmGLMSAGD5+sg4gfXdKu42Ld8u92v+B3388Ew246b4T.U.5cZ.OKTU2h6yx53kz2U81Q5+gwBZawiIn+ZCOArY8FcrTL3q5rz5nSNVcUF50Gt11D8OUpe2tAaaTf89BPlfq0q7aR3xA8HNGjyZgYxSc5o0R0y41Vgg0qI7mgWmousBCsfZFu9+dlp0fIzFWFf+vXIf3F.T7riLvkjM9U6KL5mz+L9D09uC4PV7..tq65dv+eW3E9OL52FNZ8Iexm3bui6n6NCXtENZDsiRmUP88Ic7BPHjt8TQf5j1M72sdUZECSMxeti+5mCwaOqPc1he9na+SeEA6QxI128pZ+H4ku9OYdviUF7lvZTosYalDuRJaoBA0IpJ+DJX.8tOPrZmkeNMZNOZFrXCeC6WfafjwO6LFjytbTZL+KD7bzTpQf9ATIiHivOa3sUwHWPooWkcoY.lM13NM0m6Fpb82PKDLVmkd95Klz+KtDPPLciF6yMKQ+w5o8sY7mZf4.up8nOd210cCu5W8I7rZm+abiaDm0Y8Iwk+c91t9tx+7c2dHnJpXADJQ8W8YTvjl0dRtZ5lb6NR9yAkRv.5Yiz+TaOp8Wi1Jd6VcNJ1jEpuG.H9GsrLRTvLqgU8e3CL8AgPcGy7o2RFQXzb8iQ92keFygXo3BgVcL8zhdDG8rzUHAPulaIX+w2TujiYV.V7.N.78sVM.i5MLXn+uipK8iegF3AMNs.re629g8dAmbN.vl+9ao6F.TKCGwdwewy6ObSadKKLBNjC8PC6decihYeF95qJZGbxvWWmLM7pT3u2zvXGrE2cl4nV7OGRr4j.nqYUCu90nqdLF0tj1maog0BPibv6oIpBeI6np8IqeT+.CO.DYB7RkvrZzZi.80CbkcjV6q6HWUwKlhKefccmuFidE1fIS9QLPQoScFHBUGQB+dT4VafRhE7YLwooT.M3hjeiLrpsCuNfVmdWC3sOX5jMNp8ecFQT5mlElwmty9L+KDI3zuyyA9mn+.9o1PYOO8yM7SvG6WPp+CI5WEED94FfzQcGdhaflv+i83OFN+y+7la.g+jbYokVB+ce0+N7oO6OMt+G3AX2Tj5K4vNDfWR9Wqr01rAe824.B55+CAZByYeX+wnAtR3W6eTZkTaM5PHXYcd09wj3mHkViaGAPe+E429e5m0y2tY+hiNQUSBYWBl9avYmY+rQwRy96JEdqNTOP62mTs3l.bBty4QKA.a+wruviYfOVpi9oOOKmcgAAtnYXfyrmFT.3dXA3kcHGRGsNuxi9nOB99e+sbyy62mWH6Ow2+6+8GdrAFU1+8+Eg8dO2KvB23mAIXlegqusoILc+ztoGtCZcHabVQhAWvZJUlghEolgmv.Z13rhI2IjFdiS+wAzJT5L5x6GAyTCozayJiMFqwg.ebEDglwuVMc.MZYdIdidUJ99ePCPKDugZfP4YiWTRLZXL6ryLtYxL2QqAlvxunQxQFTC8dZlCXXnm00+Y1LEyXpI+RAqovyCtACen2WwSEdl+kQ5OZMHYA4edL9mK7QMvrBHqyZ7upIVX5e.9A0+ij7qIAh48vw+sca2FtrKa39L5mXK268du3S+o+8wkdoWRv4toGZ11I4e8Cly9fLLLlgjTRJCXs5FFSAV2LO9KgeZ7Wr6W7.Mz9eZ7QNyq7QsF.g8c.g959CnzeS4V2tVyb5ffgY3f9Co.1q+498zvkbaFEXK4Ga+m7IXA4hwvOCEyVHPKi.Cu9ey1OT9II+LFzqWEFIJ+3WtPKSPxV1ND5y5yg28enG5hcB8..tiMsY7DOItf486yMmcaXSa3NrWtEKP4Pe4u7phPSSHLSNJ.17uwk3IAn8RZz7gDEplPhD11FnP0hfZPLEDPKhuooIamuVWSaGN+L4S6O.3Cpb7CpdtahN3E0rJCOF.eqVJLjWKstJN7aWJ0.u+VCrLqUGnAAn6.VwNJOKwFYT4uZiI2+0Dril8p1.0AjU5lWS6riuP5U05H9lHznkQAAjwOoPT35PyZHbmAnFdkV1STClf2Dl59KfF3kgOiexHekkcm095CJc7uxYB31kE5M4J4fxM1JdPOrEBRlKD7ZcsQFrNovYeRbbhXaYFCC3u1J+c+c+ssWO3+je4xtruINqO0YgMs4MazeH6cbVeH4JO6Y1IuIyQT+kGKX5rZ8xi+n1Zz9qYGg+3r+sNIu9DdhieU6rk5aCPikqOWOgWShBSwm8uL45YbTNtpsaKPenw+dc63e1gdQC7zLzCn5uI+JL+GkeMajhyaSSS8uhUo1BTaEPeJvFkWzmyxO.wNUaiKtfx2ybbedzVhHBNzW9gsLsWrbGaZKX827U++87984F.vis8k961zlW78Avge3GgIfznwf431MxU+MR6fJbcoqSB2fJ71DHDPt1B.pgbyvkiWqPWHP.vVa7IBuAavsd2v.6RF+pBmCua2V467LJR7uhJTPf.nAU0G0dXI18o3ZVAzk4gGDf9LE8wk6vZ.uwHGGgAVp7mQdgbpnxJimROy3+R3ug0qjchkd1P7Ce.OPrOWXdhLrOR8Qn53viH7RmFkBN4f15rC5Dc7ujaiAv29LO9YD9Mm7dTHA301bd3WCRAZ.KEcoqb4mR+iv+1WZIbNm64fsu8smoreho7vO7Ci+j+j+Xb9Wv4isu8sG3Am+QP+mcRa5ekAxeRqwcbAW+E93G820hAC0uwN5QBdElH5i4mwvuhKq+yZrL3U6G52aNN0iaboge6s9Z19i8eXlJNogPFAoAfc7uZGuYTbYg2vuOqcW941emfDeqoNWGyD9UqEjiYe3i6SRMYCHgLMTmnYT+YL9LxNfKkyTYwdsW6ENfC3.lCc2W9924cMst0stGYd+9bC.X6O3C76cG2whuO.Nri3HsOyqWRsv8v9yj3CZQ.0bdNI1rV8hDbH14Ps0SXAFN1Oasnuc.I3qKIf95hbfBsXpCQ721Pgd2UynAieOfN+ghOaLtddzt8jc3QsMW4zTc8+UpPiVmey4YN5omoCj6LHYFRzmS3li50jChwKE56Z8cGrodbE+zEdB6fEI5ZQvuWODLdaCPa8Kr53TPNosWBdq972c4mWX5O2EF4eKXWgqHsdwY3SJj8K2DBeu64IZZYgWI.D4+cD8eW24chK4RuD7Shka3FtA7I9jeBbS2zMAUNCjsenNRx81HnKv+VG7pdSp97yxAzRl079u.vo8Qy7BnjwyP82.hBzuejnajlHoIH6iwpIXbBrcbabtYHHXt1Tf5z+lX9OseIreojT.SSxD.7w2tGdzxzqC+3c+uq+m6BkbKSzO2YIhzl3076+01iaGS1IYtvmn1Qbjte1cTY6ae631W+seWKWcla..WxkbIadiadyx798b4.V0pvdt26UPKnZO28dTHMg5LcxQfRJRyJw0nu0aSYywm.Y64gH0RQU0s1KzFAzuxgq0Q+dgvgG0uqHZSPu44VCHPUja9yRv6DEGMIaPB9ib.nfMMK45cBPK0+7Jjq6NW6JO1Fj2hpGzaBQM06b3mLZon30YUn+lAQdlIEeCLoNizmQgPa+MaHiaig3mlUPF+Nw6c.Y3MimFGubvSod2D+9xDnkQY1INKDqhCveuNJYGM7aJ9sYCUVL3oQDdvqTcCyzL7IIz2ZAGvyTs353esu1WCaXCa.+jRYaaaa3b9BmK9L+4eF7nO5i1o+n4tN3HunZE5vc2.lw9I7D5+a5NZcr2e.ZVYT7Fjq9XDaneR+06y5O5gQGI0ZaoONEHt58bBvNgP5t+WJn8uhsD4AaCRaVsE58dBrlL3nNrrrj8rpcyZk0rKv1HY6nl9mFGfT6OTeAk1+KBOemfHsW9a0u111h8aNunnqoFn16GH+Ko5GreVB6w.doeL7DrXy8ejczFeo+1gsBtVsuiMsY7XOwS92rb0YYO2N29523c9jO4StvH7HN7iLEITxvkEYi9cd9zsm0dNz0nusN2klAlncT+69r1Tb3QP4AhDw0T6DADME5mGVOfMZfDEjgOAoH7FeJrgSZfOSyfyB.M5fI.lmYYUam2Z7h0BR6vNHsz96J97KdGNCKYkO8SlgpRTQzfgoUyYOLm1lgK1vGAOOSKK3.cF.rgxL9UG9J7Cbvx+d3NCHvyh67Z.7Vfm1.wH8a8kdmpqCF5+IGmv+ci9Y3wH34i9myyF4v8eyQ+iyLgS+r7ykQhRSpQMq+K1+yikUzCQv4btmCd7G+ww+bW13F2H9TepyBe2K+x4Ar.XD+a+fq+B2gBYPvbXMNSAH.O.7T+OPOKD7USGqS+cN1+3wcgRquxbJxKGPqAsa9O.JyfR6DCg5LykpOLdeP4DDc11ylOZOzxNfN.HMdP+lZmUUe8gJsweliXXN40.ALYQq5ptqOdqXu1ec423c9OyBpLrzBniPuQyVOnR+Tg2iAgSxAiICbwbiv7eE+kl8aAG9ge3Co4Qk0u9MhaaaOx+WKWcV1..drG8w9JaZSK9x.b3GwQDUvaENx3p9Xwjx4r.nJzFANSi.TUBAowUBAUoCRzVk6n3tNtvmc4PeDEcmYHzh3TaOdchT9SqfEBophpMVhl4ufAJIY4C.I9PQWGtVXn5w5gWdKde..H8ua.7tmnQBijjP+h4nFNuDbDUneHEAa1fU3BPhZmB82kG+QEftLszpa19DGX.izL7AC85LV31RMLZpGoPZK9y509Pmw6Qv6Nd0dnXVXLe9pOp4Aukw.Jf0TfwQGTzyR5+L9CetA+C9fOH9xeotW9X+XqrzRKgu5W8qh+f+fyF2+Cb+d+GS+pdZxod8KzoKJz+m3YR+ypaCdqzje1SnfgCReUWHo+ZiII8OihWF5mwkgeIwyAkR2QrTTG+J3UpchfuSmtP+KaSQgK7yZVfI+CNm4zjNpn3vDk+Nu3v6ifqSzQr22IyZOC.cmB.oa.Dr.IL4GnrcVRzuZGqf7kyRVX6vqlLg2+5WhQ9DO12WvKD629s+csw7JadyaY5VtnKZ36..srrA.bue+u2+tae8qegQ3QbjuBn6jbdFktSTP8rzYFMIw0GOyNel0mLQJGr+Da1KjyUSQ1vuqbvENKCh.acvlZF7zYvJdiFaex1nR.pi.kPivKfapNmiPkeN8S18c4igyInWzR5w0QZsqlgKdy.Zx3FNqm+2HMo7hZPYH+Cl+iG2Fkub5u3NoHAnBiZbM3HbgveT94o4rDf2XtL7kL7XH7i6+Y5GjwKc4LPneTcB6zs+AeVjL7Y8DkGhMfMKOptyCddLYD+9mKjbQaeKvFtObH9MKYPDAW00bU35ttqE+3tbu228he+ytd79pis8kgJS+Q4OrwXQdx0oCvWRvKw8ASgf2G4CCWc5UI8KENarE86A5OzlMZcR5zCX7qSHf0KrMIWAs80T7n0o6UlYEVuf0kIDPApVzu2Q+MMK6io8QDY+Mx+Rj+owZ4kCvdkxKHbw+z+p+sV7U8yFT6xuXUaO2G+LIZPF.xRSU4K+x+I2+ZvSYrrIAB5es+djG4qnm.lS4Idhm.euu2suocT8V1..t7K+xumMsosLsb0gKuvWvK.qZUG.ak.drMEKxVNwZ0BEuGEkjsIMBDaJMZgRTSLX7FbTgTYFSC9NfcVsAH5G.VvMiH8RB6X.cF6bsH3AEGL0t5fLh7fQVTa5wQIVL0Zz50SDP86485fwmJuw3SqGOqaZMPMNk.HP+nm9Y5L1N9yWH7yBfbWJM6FFdmGI3EfQQkOe3I5pCFXhC1Gcn+CKC8SdHH2zIdaD9YAdTeW3FkQ0bF+HIBHh+xbfuXvH52I8zK3Bu.r0s9PCw2OJJW1kcY3ScVep5qtWIx6czuMLzDrfkm49+4BOWJQ3Ysqr9e3YvsVFverwinkzor1kseXAz41dAbG45wUy1yShdZgn23nTQMUN0t2+C1dLO2L+j0SFn+mXgDCzDmExyrJ958cjs+CQBN3TmygMm2r9d.UO1n+H5S5Uj9gP1SaSdcf0kZcM9R6dlCeX3tfiZsG8bZs9xF13lv119Sbg6n5sCu6Nu0MrgM83aaaKLhW6Qu1XD0vixILaC94X4eldd88fn76a.cFxbV.X+qlGaQonjxmsY.yvGUXkZHZFsYyyODophKOR1ZZoggecFw14TOM6gd92aS1JVLqFETuc.qAWLIwgdS5yZMidCeo30y3Q5Xeo8Ivif04eu+RgaL8Ci9s1QYfDeOZeF.tMS3u0knH1qaD68zeoPvKQ3w.9mzc8XBiGwKUkY9zOH7i.9GCeYL9AI+Xb0I+bCjF7v+s.+a+sP0AA34+kweSSdX++i8XONN2y8bmaPGOSUd3G9gwe5e1eBN+K37wS9jOgy+jL0VJuF86iev.4GB5nld0H3SAcNW8G3xaU9ai6v3wO1ypszbk+Co+VKv5O58guZ3eIitgcD4XG+pVgscoZ67eWWgYTigswXN82jVBHuowrOMV+ykepLTcJ1jH9L1KDMYAGoS3R7qkW.eC.ZY.P.qqmwuQBv2LgFs07EoapaMqCA5uqzBvPn9H0mFQ+58gPoTVQm.f0ugMfq+J+1+16n5sCC.3IerG+yt90uwEFwG0Qcz0HsLqZpBa6+XQhFcnMpH.lC5YnsSUSQAZW1vb.gD9MGjE9GWjhe0QRdySKwQg+InJQDGBj5700bN3H0+OAxOP+v73iAM6vhNvUWJJM8dU3EqV.ZFOPP7jM3U+Cwab0c6sD826LleoAw.NpWwvOMhqnNeSF45Y9xbn+ZKurvy7nNvLr+Clyf5zibCeoldgf2CZfIOeWUOB9n0XQwC2IkB7b9D.AOnvI5zelC4C2Y1sca2F9leyu4.78LS4Ftw1w6ac2TG8XreKSa4.OSDr8Q1YrCOVd3yMGngTkndd23m7yzkMiblx320qGP+b.tHlE.+TUUq5rViHrNh8IAcauvhkG13X+LC49rgq0R1oUubEFiTivl6Ha2N3cVVLS0Zs009uPvOa1rk4U+KhziIhYYBcplnheGx.+8LiQ+8ZJ1DFYz6LqUGEOGxgdnX228cuqclWYi2wl195W+52godaGF.vW4KdN+121se6KLhOzW9KG65NuKvBLFtQulGMSov0sFaLgE60H3ZQnpwCKZTksuqelMvRQXxQhxE89.PUX.7WktJ1Mmws1uTbb3yhU61cCOB0.A0hBYuQ38o.r1LLvhoYJ.Gqk0iDnnWAEZ.JM9Qf89.feMAyxuIHcyHwIeZFLDeazOhJrpgqZCP6tbwCooq+yhvmlQjw9J9o1NiqDsEF1QzuMKj.7I5GRG9UZMR+b+O4vWjvF7If+lPk0UKZ+n0lQ9mkeY4uSy85eZKDj+Bg+.+65j8vyxuQ3u3zmx+D7W7eyWA2y8b23Yxx1111vW3Kbt3y7Y9L3G9nOZf1x3WYtt9OUop3xHM.yr9y7fW0+L3C32gWgMLFGjtpHweiZ.CNqSJsNwF9QpMizudxe3KDG+sjmX6in5Qamr.n30dmkP1ufWMxnlErM6LWxzuJ+.moTR9Q5g1r6aiec9uQh5u2vit1+J8n28+ye2+2zG.s2qR9JLcFR9CBWSPr8tl1mNv8OJkT+mRAr9CA3QcTqcHMOpr0s9vXCqe8W+hT2E4020Sb629FV30.Xm1ocx1rBUcX+BaHVjdAdtFpB5jl1IDVyJuvmA+hslJwNQMZyB5HF5RAhuNFyFHUbosXD+h29MTXfQNCi2ezJW62tVdHGgFvhf2fkHf53t1V0nLyWqNoNTkeIA4m..wfVecgNCd8Tx14eU9GkcZqnANTBqUWx.GJcv2ZTqt7uxCFxyfhs9T.fnWKxkAyRXn7OMvdROthdq1EDIBUH7CdZS6wurL8+JC32tZ7ExK4LODMFi65ONe7GclEAuOXld4egpUn0I7SvGbZ4vuzRS3y849bOicKAdG2wFwY8oNK7su7KmnUXYHJf+A5OVOm1cvuw+z5PAVMuLfoC2Y8uQ5+bJwykn7mnSJvJWsj3.N.aw0krmACbqt406Wel1WOa.+GtCCJy54gtwGL+qze19AQuE29oKjJD6S5eL3g3S7wuECdI7tjYFvbeiUlkebeBOa+N6WDt.723fXVaRLcYqzZ8Xa13Q2+idhLpk0tBV++a41tM73aW9+YQp6B896bK24c8stu6+9WXB3XN1icvsfDayUrfC.zNujfUZyfZVwVi939..1rl4nL0g8UWe7voJfiViIMO41aGPoN.wdUAGldtG3hqK3ewyRfaP1FEJTTrsn.UXrkFPaulFNaLOOq.hs7vRDe85382xT644r..n2U.sqBYMBWa.ox+M5WmkhHAigpAKw3SZrrwQ8+t1W2k1cxnIy+1uaJ.szzV3c5sDZy1n4.7BAuVWks49u.8Kb63GGoLunzA2OIj7CY3EZ1ZL8G5+U4m1Vb+ebmR67uS.pr24e+HX4FXEU6tC9f9K2oS32DcF+GrPiu+cdm3u+u+uGOcJ5w66Se1mM9AOvC3BGdoAEIh+hqWj0uPStFRsNoeLresCdr7vWn8QAUm7kjEfZ8p2gjjZ67XtvXIwgWMgg1Xd88Ah1BSFtp8f9DQzSTT6HEKBDod2+aCOntWiHasc82KtCR09qXjXKyvprfaCsOAt7CJeoxn3XFerjNzUZSVrXG+uPZ5CEOacL8qgrwKanqGo68fltUi9s2vfKyF.zxhTw+NGzjv7eAXedAu.7Rdouz4zZ8k0u9Mf+1K7y8WsH0cgB.3Id3s8u6Vt0aagIfi9XNV7bz0JxFLhPzoVYdRIMB11cordq.V8G4mMRNVJUeBlAIgveqyDZTbQEAdiw.j5PMmgn0FpgSs1TIYPX93OSAJ7Z65FmCnOvrY4mNPYxeo.AwFLa0oTnqpxRG+6utfI5uEHPgPczIWIvyg5HwMEo431n44H+Du+iClHjt0NUIx3iJ+BWRRHPm85ewbQw3WaDsenS7CxPcB+L7riiQv61iY8WhGKC1.VseLBeV+i3eyPZoqAL4R.9PsFiek2Mmi8z++8+6+CX8qewWRQtbe228gy9rOaboW5k3Wez8DfquJwwOV.UcoqOGnJqKMV9i.7xX3S5ug.B.5jqZCDBHnS9mxAyfLx4A2Ugm2Te5F.1tM7ZsoZuqn3D0LGJ5wjt3SlpKIhjvIq+oAMpvhl8adbNO5zbtRxBaBhEMlB5XUNX7u1C3SnKZmquznNsaM7Y2dbgvuQ+.1QGG.giU375i8amwgDi2+I0Mo4wbLG63JNnrzzDtka81evEs9KT..e0u54809dqeCygb6K60dsW3fO3Wl40huNf0+luYr5DTjSD.+75KMkBeWzSC1fhKeEvUkNMpQ++FG3LQAZ.ftTjYgsBI.Oie.UAQ4QXJsA7qJjjyQyc3vFvUqMAHyBE82DTuXfJoL.voSxmwOK2EwurfzMDXbBzhIiC9M49OJPAXxeE+il8UcVEE54twEl.RmiZEm9G8npUbpx3F8Xvq3GNbd8GrwbT4qBm43rBS98oPTtvzuOyYFmyGdBbita3lrvx8ew9kHaDl8uASr+Oz.Y3M4Go+h73WpOiZKabbiF97e9O+J9VB7a9stLbVm0YgMu4M04fOJ+h8eEh+c82T.pZZZSNx8vFhzeT+aYfuLF9tYrW7wG0g7jhm1NlNKE3EEbg9H+MhmiC+UBtmA.8Yl7q89DQS+bAskCrk1e8V+yveg9moX.yQcf7MQl3zbQ+Z79JH9F8yokbF.3.Dp16i1+U9t921o7ZJ+tkw5orwOpwTRbqbs6B.dFCz8OgV+vEKWGtfgKkAj.hDO.ipvDyPAG2w8JmSK0W17l2Bd3G7e5ucQq+BE...vsdy2xVVIGGvi4XOV2nE4bzbD1bhGVyb1pCPmxcb8ajz.EMpLqGKXn1pGaIaPw2Df0NbevSgVxAX7C2K6ouQ+rans3BfdG7vLmFfw.TkelzfBnhzep1fZp3xTKXzhcr.4S9hFfisLG.1RCnoJaH8ykQy9PMzJlKOO3FDE+79EH.O0NrwONPKmpSxejfmfIZbmlUB7pkcnFPYZ1JgTD2pnxeliB04.w+NCjN8uI52vuBEOyoX3BD9kgs0H72I+xATrrv2ieU9IVag.7Ze4+z+z+D9hewc3QTF.s2de+o+I3BN+yGae6OYD+ij+C6+h3et7uXR5f9C6bBTcxiIBCGG.+bwuFPfRyz3mfNegxVV0aWu9CEjgzpi5zy1nez202C.yJASM1QGt1t08fUoLKh+ffn0my1bUmyppBMihloogze.bFMCL+Dj.F7dle.ni93rYCt6+YZpQ+zmUeVF4q12MGzdIdoxkybSpvBflM3NSVBvtu66Nd4G1h+5+8lukaE24VV+u4hV+EN.fGcaO9e3sdqeuElPNli6UZJZg2s1JiCXcVpxzvKpGDUR3asNwNtEQgs14PdZLsHsSU+FWpG2P8Bxn1.5ZkqAD3Q4Kg2BZhBiF7FosxANnJStyNdV0dfLFLZaQ0MHPHqxVxCfXQsW2Uu0jlM0DKynzSqqGn9RAwjETVaHzaQuaFYLQYLM8rwxvRdzI+79yZvfU3r2Jjof.5l8m9bCOyGdklsn20fUx5EVc7YkyF18f0lbbnc.DOpOw6IvG...B.IQTPTwVOX3XH7l8bi9Q.+zZnh3reT7KF9Y3icfRh13YEWe9NF9tkGvnep+WaKUoYf96UbkWIt1q8ZvxUz2de27McSyA+1A00oYUWn8IWt.SmQ4eK03pbk6+nrFw5Br9fmUHgv47gGI3Cm2dcbdggAF8a7npOAxYOgyv3ulNC632smAa7eo4rzsSIsqHbo0VyZ5eSHvnYGxpEbwoYabUyVoANq+zvoNgPuqr1BVvQJMCdITbCv53e19MT9cZvamOH1+ykeEusrwkdFTq54t8CdBSylMyR65vM6rR+E+yZGnmU.uNSPvQczGMdNOmmyv1ZT4Vusu2Sd4W9keOKZ8W3..9atfy826Vu0acgIjCXUqBG3ppu2hs0vwTxTEq3NcT6PBOgTRlll72fcv6vU6lslHgKDz5rSk.Xfz1W7SCPKrU8nGZ68Uhl4Tb65qBnQAPipmML5ynp3TgNntx.jgBwb54rgjIck0azlZDudz.02O.5941hxNktr50koaTzn.Z.Ygfmc3CDSSu3dz7fyH9nS5SF0Xd0M5OXW05Q73s+x.uBCS+9eKpMJn8DQmv.lySS1j5DR32jIjPItqf4PKPzPPNPGq2iRWLREstF7JIkj+4.VLv2wvq7+H7G6q7AfN7Hn+bdm+4iG5g5Wtx5w66KTe688HOh2+kveu7KGvFK+Pm9SzH8bNFehzieCjR3uF7TcGB+H7awHQy5u3va3ct5uZcsOB0YnFbul5+33em208LjIAKEf1F+ywOI.3A7fsUgfcXg.LZ+HO9AV.CFuR78nwOr2BQ3KgGgdymNu27ek3+kzUr.snwDRimzuq9DpKaZioYbLHH.1mWWPBF+WqyLTvq73dUcsw7J228e+3N1zlurEF.rBB...Xc27s8fKszRKb8O9W8IPcV.pyDKnslPl6PGkEfjYp5yH3B1SgZffwEWOm.xAanQuMEVqHgtjLHODD71L4ThvFHFwPnNvoeOSBEe.hvjtGfQbLW6AEpUaAXYQrm4Q6Ikv8Tsd1fA.klP+uQ5GA5WiD2hAqoHaNDU7Y+H0dkRHS.N+Gmwp5IMNf2cHBROKCuKqjd7COS.EBdYD9R+ED+2QWRl+cGDifOzqRzdPlS5xNbyGdw9ON3co4dNz+bgOym.gLgYvOG4G+8G6G9Cw4bNmSvFwFti5auuu829a4rb6SN71.84R+prmFdLG8Gp9I5LpuGmsc.+SSc5ecv2M1c.9Y6WbPKD82MdBvVhOSzHNGTH5xdyeRiyqM6TvFqdhgFp+EHHzL2Qx4hFfpfrTHz+0rSwADDjK1+p.UBsh+4.7E+HMJRcBMy6L+6vmlPnMQLG8dn9D+HNdWRhYWv5+jLt6Gmx+UWRC8667NuyXsG8he7+tw0cyXoG6I+vKL.XEF.vOXqO3WXCabiKb8O9W8I3o+sYrtjUd.7Te0IvPLJPxAsHvRShVDpyl2aAMBv7P55wofMncNGuUNJRTISUPxY.PfiCe4tZ+B6foYMHTmBIibwiMya8jHDLFDdPTeqRFyfdycwoar.wVZ.fZjx7YAlKid+AnFA0zbVx0QIqRwmcjx+bpQ01QeNEDPXF6lD2YTFdrCfORaQ7mMbkoeEGg6Ec1fcNvCtncfjAcUBJ8DvP8esJij+Vp0mC7tbPhxF0.o1Vyg9GAuhZF+g9eNXT6uyCdfa8VuU70+5ec+s22YWe68o8abaaCmM4ewXfNGIMcuf7KLSch+a0ko+H+C+dh.Q4OCOqKTxv2bxxvOD+b6Q02neN05V6T55Kcw.mEy1yDwxvYUTniiSi+CxOloBcLNeQ5XQ92W5JyNox+hVmhySTKnCi464BnQSoAA4DP8YEX+S22XiCDnjzKaNjqX1cHqMcP.3xV8DpY3f5+r1NEJq6qvax76dgi8XONrK6xtLftGW9d295mt3K9K7cWX.vJL.f0eiWy+62zMu3GGvC3.N.rpCXUUCpkAGcINfJ6GRKC.a3pkhd9mJNXIYt3gvAc.p9b8myFlgkBG+LiJ1UP7jRyvUj8r.nLhyAhPgJXo9Qc341wX9jUU7euXqinxBc7Hy6JK1ReG.niEHnuKVFqX4uKeo9KF+zuCZPC8CtQKJ5V1vWW6SFTMGDL7vgmXSRtSv2jKA7qFKxNrx8eA4m2+l2.XJJMC3rCAt+mMpS8eEC9w3GMmlRh+UcHS9E3kd38.EU7m6eI42B.eNfmNs1FQZxukE9JbekK9qfe+e++i3RtzKAZ1uz9OOY+kv2c7W6OOpi5nv+pS6zHxOBeD+jLW7OWH52f2ze8SsTP+YNxuH+yvmEyI7aBFwpeAt9mqzk6+QWwNFbp8WATlMK16Gj52nw+FO4GT3ho.ZhO.p2wbPJT0r+iu19ropftr8Wazi6uv7eP5Y5m6D.59Znx60kMVesxGKttQ6yDOJz3O2dcLqF7XNcy+YzRpCw3eGkFdnCXIKbwq5U+p6n44Ud3G4QvsdK25Muv.zJqrk.Xcq6Qtoa5l+g1NMeAJu5S3DMkY93MnBRdskCSngKIgY3nVLXcL6AWrFtZmNFE2nRMpN.fRaSxnQPW0H8NROLNMJXO5WJPglhrYNmrKYOszwptrp4fARNPmx.AFGqZsQ8jaTU4TG+yZQwV4YDLRHkRKKKHh+.h30gSind7tiOndq0qUw3yyBfTpbIigc2oBjAR1fJYdKfGKv.kWFMv03eWNH1umZexgs924Cuu2X.AenEJw12z+B3mEUoz1pzjg+nLIDXyB.etez4OkTSPWVN3q7+RKscrksrkD+6MfZLlmwuVmccW1UbFmw6F+x+xe.b5u0SGG5gdnIpOF3gpcx844fZrmSzpPOyv+.8ONnxwvKAYV2LEI7y1D4rywfXa1slY.ciow7uNi+RAsK3LuGHaKIOSYOPUC84V24WytNWUcCfS5+r9aJKNP3r.vD.EfF25oTJqAhnmMe8Z4sOC.RmxrZq1MoVb0CgoURCNL.br+mlG.iVY8urSeE7ccW20Uzs+2MbCqCOx1W5+iEFfVYEE...v8dO22WbiabyKb8eUG+I.UbFRwr1QYiM4c+4fn0nnoAjgG2BGdpM0.NDfv.cyS1fr..OhN+ZHlmMrw.P6bMbVTdgnIkNXC8Dc4ShvU93ALAmHLtTAYdZEjMMelYSg8mhRe9w8o0WT32S.vB7wBBHAez3ZLZ4Rpd4zvqjabVnEN9nlbnXvyvoFR5lEK4H2ESQ4TvfMwqfqO8rb1qhO2wuNDm4+RG7RP9xsUS7AQlB8iF8yeOg+P1BR8IJ9Y9VqGqqtif2Zg4g+Vjm5m83NhxOD3+3yBxMxgQH3.QvZV8ZvG9C+QvIexmbEtRAu2+GeuX22scqSl3ibb92BtJqqTJcvGzeUZkxJfAeHc9k.7HCepcMtOieyj4fL9HQ4udOdHhdDl89Vt+aJPVts.e4Llf4PMGwG79EytHYqx52JjMgtN.3AbFz+psfCG2+6z.yy9q2XwR8utDG4Wo7Aweiuh1pS5+Z8XouZ2tEPEGbQ9nM6L57suzoqg5ko2Nuy6bGcOux5V2MKWxE94+RKL.sxJN.fq6p+V+Z2v5V2BW+CXUqBGzAsZ.PQqIiq6f4oEJJ7QANnMoG.G7.mdEyyD3AOcfDJSs2Dg5Ykue8nFPi7nES4If9H9Kwer.u98AlzFbviDFkAiL5aqw1LB+5m8wF0QnZVNzSBgjhv1n+RI7LuJkfxcX8vxqGO0OaRMp4539t0yORYlAXEdJffw3OQ9lC7H7db6o1M0.ryoQ0qCdR+P+NaqcX2+xf+d3YDzieW+vQ3JA9L9MoULFg4Ceh+y9Z3zhWn+A.7VeqmN9M9M9Mv9tu6KiD77e9Oe7y8ycFA9ISoci+MBPFq+pvOR+gG2lBDzBdMU2n9KK.7fRFheh9GYxZpRjTy4q6uXCCKg8JUPDn1Enf6BEyI9.7KT+u9WlNoNu33uBZw7CSANLImRnpl7o3hOOSs9oYBHkV9LwRvGvO3SvSj+M4udyhNe2.cPwlAiYyg0p85bhmzIsHMN..d3G9QvMea21h6TlJq3..V+5W+CcC23M+CWZtB29xIdRuFKRGynZQioxU1yqSRtvAQpKCvR7fEDinxATZJnsVn3li7.rmeuoHw2CAybFHBeQwOPvvn9LQHkWSafn+3LJTm2hnitT4UwkepxDaTIpSAo8F7xNliE.8zzpACLIn8RApOvCQD6pCVWUP0PsmgAR9qxGl+a0I5TMx2ZPOL9KD77r0Kb6UF.eF+MXCyRFDuRsyH9uRCI52vEw+XQfuzgeoC+8zuHj7lxJhCOhvWxvSzaC9o4x+yCdzge0Y2n9ug3uAux+JbXGAODru669hOzG5Cg25a8sN2WrKG+we733O9i2gm5qb6OT1iRN8yANx5egfZEA5l6iamNKIJ7XD77w9KsOCX7SigxxOsslAcc8E5B9pX29eB0emmLvT6Z.bZZJ53SUTMaJ7WnY+6VTFz+Ay1mZZTWEhP+OsOI3.GjVzErsI0WAeSZpci1NWPeOnGR+eDd9jPo1gnI52nS21u95E1lL3zj8h+wZm4p+6iSiACzF+zr+tG6wdfWwq3nvhVtgabc3wjs8wWX.nxJN...f6+9tuy+N13lV35eBm3IXWlAdTzjupjCKsLXnjUMM5Ncs5sr.v+0LX1h1Rk7Cbx0UnSbftKOEtiO.u1aF4CUw0UX7Ff4MarZpAbEdJBQzK+5DNNADG6RZtUCJonOSxD9EEhdRA72m3rQhgcdZLKgRdVTNu2U0nwvDtxARLD9ZECvUZJbC6+6dl2KoZPiwenSO.8X36+kd76As3CR39+Qzeh1kd4Da3si96v+.3MRMi+xv9Orvv20bAoDJEbJm7ohO7G9iXYTb4J+r+r+bXu268tSOMa3cjcG8wKh92PcesjbFXiAk4f+433meF+ac5dod.90aN.pigK0LApneVPwH0dRpIEB+CmvTLvACbJXDu+OCaEfH2jj+kj8ihO6eofzdeP7q8zLdPIH+CxQyNbLfCs31+zGTsKNes.ws8CX6ifZ61Gn..vIbhm3J5x+4FV2MKe0uvW3qrv.PkmRA.ba2z07qc8qfkAXu1q8FG4qPeEAGm0hfXTbVFA.k9dsPvV2bG59.ftHFDXsUqgTfqse61CrFsn2Q6g51JIEmIHVTzJ970WSb3aJlN8SJSVXrV7y.sHQKbczmEFSRyBo0.1LSznLXcpzeqNCqmJfooolQ.wjK90DrSugzGRQv5urfR7OYvSCu.vmgiR+gY6nOqDg2LFxc+q.38YlBq8BQgSzsGENI1XdJi6g326OVN38YQGWm2hgeuCKS+l9ef9I34wOlwzHuozHGFRj+WN3iyhN+Wd7Ijn7OfeFNs+nzS+p7aO1i8.efe4O.Niy3LV3iE0y849bw68889ru63N0GKQ9Oq+3Mf22Gl0p9279AP.385h8rFO1AOhqy87ven+qM9U2CO9a5OXW5O76zDYhxDCgeMqfPlhx+QwVDneiPL52dn3i68PFZ59ZUR3x58Kp92X6+Z8409W4U8yS.iW6enYgAo9pF8niMCQi3siO9E1a8uoPVDx3igmyNhuQxYechH3DeMmbW6LuxV25VwscqeuaXgAHUdJE.v5V25djq+FtwGZkbo.oLUb195kOp0ii34urWfpoty2G.8B+.7lhVghfjN+nCfOW7aWp1w4KqWQJSnXWnpcNy4HME9+nv2nasAXio7LXIx1ZwvLJD5GfIZaOqOdUgbFMqTr6Gba1ehXOyBJgk2.vNYHL+SFy7Yv1WhzOMKACUQbsvvSFgzreLxj.mQfRqMX7p143eqqUz0Er.CWlgFhNMnMdphe9bNSlKhUcH9WF3MCyr7Kq+w0cGC+P7Svy+LGL+7guU2ow3esqcs3i8Q+X3nNpEOknZ4vd4GFdiuw2n4DJH+oRV+YT+edF3Ngmren5XkD7FxxmPFzBTfF+B8YDJJt8mXap2gGsIC0F+pv4iic6GrLdp01K01ry1UwFSjrvKP+jCyf7SeFnwVL9SBkhOVk6i5CFo31GQ7k0lY+hagAm8eC9T.Nxf558etAzk6J9s6Yo.8pv6+JSSZcNfC3.vZV8ZFhiQkq4Zud7C25i8+5BCPp7TJ...fG5A9A+Q27sr3WMvG2wdbXW1scyi.D9U9Hml97rzxEdFPyl4uVK8T.4yJVUlMmwhFEp.pOEouDJyrkBXx1G.55gaWOvhPJ1Zz0TizhuQDk9D0tsAi5ovyRP8YFGDhbWo4FLF4SegUxM4L.n8CfFfbAdLnShXGaHMs+d+hDlkASWbp4r9O0YHI.DtOLsbPQ4Wxf77fWwefNQn8Y5Wn1wBjJPKoeOuNrTaDx5.wKQZx0QrLDjveHPEVGEwfg3Yf6Y8.87upGjT.2gzOg+wv6zNX7CJCFBn9O5xtNI6Umx13eesIf.A6xNuy3c2Nde6wdrG3oZ4m9s+1wAdfGnQSL+Obe.vzH6rKo6o+t12DpGY+JBuP8Cv3Yu+2Efw.Rc3s9t.7vtpe0+w2rm.0.Dqiu0YNSuhvkI3WkYZ+S6ezmMd2LO4vTBJ.7LcgGfiQ50Fj4YNvTo8oVHOQ9OYeDJJaDfsrGCN1eBKC01twr9Q7SYVU60uLvbafz0Kr1+OHv.y1cQwOLcC17jKqDbRqfY+C.b8W2MtzW8qddesUDPT4ob..ewK3y+ac8W+Mtv0eW1kcAmzI11YiAGhvUzruF7HDpPo3QAqqOOu1OhoPwFLicl1qRXkX73dMkSsL0VGI8EET8XlnmcdlNIVg7nwobxhhDL9avPNsciKBDYJXLP8oZ5pzf7pqbPFHvfRA0WYv9k.jcjfTRnQPERo2dMACDdehOdFcQmSwYfByvazISV9QvKKC7cyHibnPzuI9o.Jli5mUe1oQHCFAGGI3I7EfmKI7m4ey3J6vpg8Bh+1nVfrMFfWclnzFFP+N96g2Heh25QuP8eRO9Y3omApFGzAsZ7Q9HeTbJmxoLf+VYkcdm1Yblm4Y5qoZItALyZugQ+cxeDj+AAU.9b.oQ9mC1I.eY4gOfeC9ZQey9U22Ns5J0f0q25mDZJk9kG27FAuiIXCV+ptOJLCUC3eZXoH1snn1H1sqWO3A4poi09dXLLQWDS3q6eW5+0LKq1ODyOSP+mTeAbm0.dFPm0vyxeMCS19094hZ9wCbwntBvrYOGbxm7hqyeW20ciaeCa7+2EFfAkmxA...bs2vMdGO1i8XKb8esu1WW8Ch6D18gH1y0Hy7tbuyLOfPuomza+INE+bDXpicO57nQTQlPRMyKzkIgNnRZ1.Bmg9D7rAOqBPoE0Am5zkkEfzB0c7uab1cv3AWnNZCG6QYv+ZAUXunfzSEf5jzFPHdpDK9KUD9EFzDX9mlUaq6LSJlXH4fz5WSR9x7XkL7EB+cxO3NraOmmwm0VoQ9Lsn+N2FFaNsfvyvlv+P9mBhrPvYswbgmLjOD+tClfrbT+WBdE+13uD9CJvI5O2+o0IrKtAvoc5mN9f+u8A6NdeOcJG3A9hw67c7NHdvy3hd126bRCDB5i4es+uxK98QBDOSXbak4es+Kq+617FCOX3ALm8ZZ9sW1OZWP6u9l9y6q0i6mXm2+oz3G3JwgILo84tSzhSRfmIu0+yNYI8OHQ9uhe09oDjDhZ613+Rf+02po1ICI4bNCuM6eQO8Ys9+ANmqxZoc6BRu4+HQznfg4KINs+UDtt73Gf0dzGC16m+yuqclW4ZttqC2x1ej2+BCvfxSq..d7s9He7qaEjEfCZ0qFuzW5Ks8M2bCaXvRSCKb5hzqUlwCQzNtZqoGGF+Y.wg2MiUbnohCevBXSYp1o6OuPoHugHJPCMyCJDk3.AHAxwcnaMkanwLNEBonOTEQkU1H9.aGGHNogiV4sBRxLXKCf5v2U4A88D+KpzS5neOOgjsEtMImrxf9uQuVUcg0.XFXbWXXRqmZF9kC+lAJdYFn5NDdsNI7GjefgWaWR9kCBH.dNffw3OJ+JAYR.+nWVx7eO9cC4A5ejy0F7pif8ae2O7A+feHb5m17OdeOcJug2vaDG9gcXTPPfvepOi9af9SYtYg6+45aU1uzrb3y8+CfW7a5uvoeBvBRG.0.SowxU8NWCXRpKuYm8CFYVrbt8m5GK12c5SBzOL4WI1lA6mZaTTunl8WBbKfgRJRBcIX8y8+nW4uv3b9U7qN6eiwINPwo9YaSFFj0Bvr783GiwXaQlZoRIHmO0W6oNnkFWVZZBW60ste3sbQWzctv.Mn7zZj1W4qbde9q6FtwE+BA..mxoVyBfGIj0i3QOKrytAmF.zFfrT615CzMCXo8NulBkkUbzNCQpCRTid5Lo8pR3zWrrVZezziOQ9YEnuOtEn7hOqJang3Q1xQMJLO2jE13FJm+lRk3Cr0+iFOiXFmE2pu2DD9amJ.Q2IvEnmlWUFkCBPi9uPFn.D+RHQC1.D9nTlpFd0A6rCIdDxhBeLffh8a7xDDeOJzm8AP0K1+Eg2vYq+XhoIwWlAEdixlG9mlrm47eoilsqNTgxXQi9Y36x9gzieN6W7agQOfhxP3cSjNS53mLRORlmweB9S4TNE7Q9HeDr5UuZ7ipRoTv68899vtsq65P8Oil0Y8SYMh4Ee7LFn+47uKy7mwx+wvWBviAvOAXK6oNq+5te2MlWDAkYdZ9qCK88.fN6+klrywjolXHTfYfInKaFPzwejLFd.cdV2z8GfZ+v702dKF1ZSU92jMgLDCMCq98PBLYYsLgw22+JkIj8Sd1+ZeJX+LBWG8lQUwiXKAfdpJ5KQ3462Fw3S3xOAXud96MNp0dLyg96Keuu25w8d+26+0EFf4TdZGp8sdK2523dt26cgq+I8ZNIrS6zNA.2HaWjQTjhdfB8EE9Y.lSZ8syj4vmfWDJcOEXJbEkFDUAMEWm9BBpsW.TkPcmwqEwfGAinlyI0v.YX2viwyRHpWN3jP6XvP3GD8a32M3ZABvzjJ5JzkHBJlgiIAscHqdZFni+Xo.98HfG1tSdN8qKKC4DhBLqrhgW6zXgPrttLgTFlW5da0ScvxFlCsYn+aj7Od7mL8uRoC+Cy.fgWGlvLAWV5mzZUcFIx+r7qybIgqL7E+Qyg+KIYUO8GjI.XO2i8.efOvG.mwY7tWQu0ydpV1m8Yev64m+m25aCzu1uln0XPeoiVZNCH.o5RGsOF9A8+YccMRqL77NdWR32tveZArOq432FuA8R+YpgyD86UinOIz+aJE8revVk53Sso5A1BqNQ76N4UYguzC9lLTO909lbzsWLALdy+o1TMmwFCplacFW7OWZsucECq32jeH7WlW7FNFjfw+v8eA.bJm7ofmyJHyWW80bM3hNuO2uwBCvbJOsC.3Qen668dkW0Uuv0e228mKNgS7DAfKH51LEA+.xHILJjgvkR6nEQ3TAk2nO9feM9O9shkBeVMhuXfTbwyJdRaTPFhaJz5Wzm4t9qi372m5wHC45Cp8bcTJHASvPzeNvoT7Ft7q99LWu3rpC3k1.FdCDwunQhmic9ZH0CjHUxOybxvy.1cD2wCI3EBd1nt88T.E41yneclPKK96aSgZmLcMB9Q7iQGI5Rj4HuFQ+smWX30lOw+A5T+6zT54ig2nGyGYa76BR+F9k1w66i8wvZOp0hebVd0G+qFGe6srlRaF8qY.h5+XyPL8C5uRa7Sm7W0Y7guggyb+uDdbKqjI34T7Cv1el5Nu+SgtDcboZ5pXAo4N1c9z+7fwvTU7wO7DZJQ5mL33eqMNCb.QRKn.UuC1PtHlqOvuxeK14wejyLU7URvqSxBENSDNeopDZlUlkgm4+jcVW+GdPUZv.o5okW2q6mZ.0Ot7vOxifq4ZuwaG.K94veNkm1A.bIWxkr4q5Zu9st8su8EFlW+q+MV+fMoHxwWq3oiRihrWHGdezSWLP.sr.zlYqjf2Fuqn.55y6OraYGr8AfML0lAr9lmZRX3isCiOwbtV+ldTWrzqCuMznP8nlALqyEO0TgLCX3uo4Q9vzG2.2qawkeUVTSgmlJQ2DU7XP5OWqic2bSxg.uYC3hDRnNE+DaH5uQok2vpI.Fb69Qoz0lYNsz.7r54.h.8LKUlh+bE9vjkLZjvM8WC+JLZcGQ+EpY.b9e.8ub7uR+ff2k+T7fCF+kgm0sy3OzG2o+S+FP8388teO3C7K+AvdrG6I9mix64c+dvd+7e9lrHSirdi+a9r97f.b8GW9m15xTVPb+.rmAun53ifeRjfiH6kcCP5d6n1lyJpoyJEqW9WyT6b1UXLgrjIOaL.R5FRaTeaZ4Eyw8jo6FneD4ecRQEV9gAieMbv7Vw1LdVF.ZL1nW7O50qKKaz.q3Matu+FhxCM8+KQzlPs8vB833M+mamf80bLG6wgWvK7ENtsFTt5q9ZwS7nO5+yKL.KS4YjcayO399Aex0cS2xBW+CZ0qFG7K6kEh1ziMRLGjVJ4JtxFW3YNwWjCSMENcM9KD7wHvb6mdp+HZnaNSwHv4LMHhlFbma71fsU6a9GQnVWX9um9TG7ApQb5qm9I7m0SEG7JXEf1ICP2O.U5Tzgn1lDjeAXvohqPzusQjz9OnNxEhe.IcFv+MXCFNBBLONHF9F6aNMC3W+NAynn3Y7qsazvObC9Y3EpdA38kNv6yXRo2.af9g+LtOuCdQFR+95vS7I09Hg+.7hXFR45IT6YsqBat8KEr5mAOdeOcJ69tu63W7L+EixO9uMdl+cae.jsY0I+Dqecj9Sv9CMdfamQvq2tcr8GcmoCDG+UPao6L4ewt4TmBYJcfvwlggBuI.53fd8uhq6psuETDeRoHXrwsblZ8mw5O7UarZyYRm4e9LMqv2BPQy3gieMfi9w+Y6CRy+Q90J7n.VbC.MJP8eo7ei.TwrHBd8ug2vf1Y9kq95ttseQWzW3+1JBn4TdFI.fK5B+b+6upq9ZFIMla4m50+5AzNmjSwV+kI3pxTsqKhF+crre8.q6XyRqQUGupBJqjz055s...f.PRDEDUyAwYNYP27+8RKJScfXNJvvLsr+1lwewvpwF79.fmPGSTlSiRIvFBy.JuhA3m85lXLKCCFCLEEJPuffzfqkvZOpGAS8y8CJ7HvMdkVidl9JL7dJfLCMprzDThWMUVLR96ARVBhQAw9ug32bb53Qo4f7OCOUh3u1BA92Z.NHgPixlcmO7ine8Y5.K13lx+DuHC3K0YjBN2L9FTbN7eqYdqm1aEevO3GD629secxm+4nbXG1gg2za5M6xOQF1+2o+.JyOhWunivXI31L86QyG5smWrekc7meecTJfBJujadKv8513YVi9oS9QPmnm9hQtnyX2gIjuxhaIgGUDlHho+2bL5FEqjvH4SC+18veimqa7uYdPQ4Y+asIr92dFH5.nKSys.MJhew+LWeCHJ939behXQ4+pV0pvQbDGwxzhwxF13cfsroMcAKL.6fxyXm2labcq6JefG3Grv0+3O9S.64dtWzjjXEA+pMMLyJczAU3YY3QmEWe55eiJmbf6bczl2L3MuvZn8AfgejC5v4sPz9.lBAGkMmle92izoqHaNhxQQWJQdxDffsbSsqVkI6GTYo1OTf36pXAVFW3KdjZzxERt21vflSrbeo6bw4+H7B4vedyXimEqGHoaP2MXSB.52mO9isM2AZozOCO86b8j486Cvk4HhCHTwurLvC3zu8dxf2LoQGUC4eF+I9ONKI+2ixOD979tu6G9M+P+l3zO8S+GIGuumNk2467chC7.ewc8U7my5DrQc8ucxe1QGom3clwwm732f7CEaMnUmP8v62kBShmyRcwQMahsS7CTJkG.kBBQG2wAmGxloRq5ySi+h5uL+SngsYDr+FwuaaskMPQyFhX+cdy9OhmnsbYH9oiqWS966yfka1+9m0ItJflvSbFaAB3m50+FP8hYawJW4UdU3Kt4M79WX.1AkmwFQt4G8Ae2emu6Urv0em1ocBu1W2qK7LKcMpvKobN20bgJ0nzzAentNQs+4qGO04U3OvG4POZ7thte.Z0yO9Md6XCxzIRhhErIGTSD+9i05oj+boehhMBdX66xgPo3+wLfYFT3fpKzaWSG+7YQNr7HEePJZsqYbkGvn3j3aW7E2.m86UDVNvv26PmYV1fM+qwVKKmI5jM3mgOOy6TFBF1tDcSyMoO3yr9y.36neJSJY72odGvuzK+FsWAneVRO3jeMmB9He3ezd79d5T1ocZmv6+8+9q2RfTVozR2rQQL3Hafo9e4wcPcnWRPOe4m2N0OG2ZlkTv0MbDzuzc9eC9V8lYzoB+XbCZVpg5EfNheh6s9+f8Ky3kSXQ6eEaROA5vve8G0r5xuiRVth4Ce.CDs+zo.CQ32Dp0xrYE+p+cNCLkb+O4.ov3u87ccW2U7ZVA27eO7i7H3ptlqcC3a+sW7aeucP4Yr..th+l+l0+cuxq4Aexm7IWXX9od8uQ6J5TSQbNRM823+MpvYAfiTaFPaWwxvyWmkdjpZj0YEbN5Wug0nB8npUmdlh6.5WwWvYNgeU.35K9ZMywfaYAmh91VmqN7CaFv1rz5jw5flojgKMEhQ4eYV8nFsj8lvR2vL88QN7N8podle6CpFW4YBYxORN58KI4V3YQ3M4mY7t3xC3NCGQ+LteJAugeDn+.7Z+2bfWxxugvOe7OO521bec32rVYxeL.dNCCJ++7ddOO7q7A9Uv6487dvttq6J9I4xAdfGH9W+u9mg3iw5OY4eW+OK+xAlZOqM1r35+dlF75luLazmYuQNavWDAhcSdBHswegWU4k5xhpVOrt1rMfP+uBKB1Th5ekfrx5+Uapo.N4w.Z7.5uYAU0DD5G0ha6vGuaWA7KyKwGkOEcxXR52Pr+S2mKg2yK.1dmXZtm6+F+Sa3PNvh.+Svepu1WG18ce2G1diJe2q3pvS7PO5Ytv.r.kmQyI2i7.22G8ptlqagq+du26MNA88C..xdkzNNaS7o+cAK1Kumhep.rMCnZPdYaAevPGV0oGOibUKnaSx4zeqURQwaPyg9qXWznU4vAT5Rhgk2CtOXzpSoaM23oA1IKjI6pBtt6g0qSDXKAPbY2hGEG8kRhs6kaFTx7e7CL+2aLf2UsKGrJ+zMK7QyhcNE0oWCv4NKnP82A3e3rn01cNANPJPA366uxeU5jy4rPwZUc3GA0iP6LZVPB.V6QsV7w+3+VXsq8GuGuumNk23a7MhC6vO7w8ePGeVFp+BPxGIlwJyqER54x.42ffx8z9SuTtR8e4lHx.dZ+C1YjDP52S7j53hoc+sQnDZiQCAM12lvybNUAtm5Pwdo6z3awvuDNwWcEGI8iyneypXa3kM8p5.+tW5OU3mu+mHt7f9La+p82Fc7FeSu44yCoxRSS3Jtxq9w+RW748MVXfVfxynA.bgeoy6O9JthqZEc1Deyuk2hsNwbDlihhxlMxHk+zFnJdL0T3YCmUEN0vaMf7TD8M3GL1rgDo8RgXB9HIwNdJQ7SNjEX3y4U6m..s7CFhU5mqi+Lsx4Hx0eABOCXiwQnRbDQBySvh9kOY.luQQrTyIBuw.4aPP57VKQ3Cxb+gHexOLGpMAXtux5+ZQh2aPz0erYcPDhDnsT+NAq00DvOh7Fh32ncxodFdI0VY7ikA9BRzehBDp8Y5mwY3ypgehdT8mX+U8u6b6s22uxuxuB1y87edNdeOUKkRA+Rm4uH18ca25j+Vgs+.p+2FGGOtn0GoiTneS0e0w3vk45opwPWX4j7K5GcrECu.Zy9AcO7P3WBnNYPipK0lw4jv1eIhBrdXxtI0.dFHZFvx5+MGvJ3U6HtwI6cvhH9s92bdi+YjVdRSZ62Q2JRamrBF+g8U.M9Lg2BgEk94iVHO1YBBdkupWMdgqfi925toaF28ccu+9KL.KX4Y7ckyFu8M9k23croEt9u3W7KAq8UrV58odxoOT6cZDTXtyVDoeRcL6vCD8o3ACnoB0yzPB+s52Ul4GwF.z8Jz0wuBvnVpPa1t.S4NpAQ+sVfaRi+K7dMfniBOXHQ.7mon4USc5MfHS+ZMzKVD61Bj9Y6FSj4+RD9ryzvdEH6fMP+QInzNWy7KCoQvG1uAz59tb3G6.3yzORvqFSK7u6VQCFC2Q7+7fepC+jANPfOGM4Q3OrQbSYffKqd0qFerO5GCm5ot32k4+jVYe1m8AmxobJ8S5.i0+X+h05LEq.q+gA8eomwmyeaFuTfAV29.30T+qYpSuYOCokOGnu1FhWW62ayXU8gpAyT+JYfvvdz9i8Kj2bUtxW26N+SiQX6OD+KMZRCJX77+qyvte7ixXNukfxpaN0+yl4ajxRh+3VvumHJFJDwClP2CXZ1neKuk2xPNXdkq3JtJ4htvO6GaEAzBT1omoavu3cd6u2i+x+N+v0r5CZgg4M8VdKXcqq9REpJz3NKDFLoZvRgL1SEyU4TL0MQ3qCJ7WNDoOC2.u5QjsuJTDeXhhHE0nTmJ9EDjqLnnu4zqEThZhfMLqCGKDGZCPEk+0eUZ9.nKHFjjeDAH1.YQYFWv09qzAtOHvdQjTHYlH1QDrH.SMbDdYkzLBnqu4LwwOYeAPp7lE0fRiL8q5HkXGiIKYaqMVSZ7lIaE+hjhjXgYvDbFqsGMCP9RJJ7tG.8kfS7Fu4GAOw9LTYVSnDneUXIDsQ5rJdLviDPnq1BFInXGwuGvBuTBj7a1LbZm1ogS6e0o4utceVXYaaaa37O+yCeiuw2nOfN.Xbs+isG25+zwepRFbGplsDwG2KV+O7YzB+ssoczyZXWOdshnaB21yIZonOmBLL4mN472oNcdFAUTl9KfzeD1XlAOKKL7oAKH9YtWCVIN7UOZbjLF5ELTw1+ClSeV.DxBfieIDzD7LO.pukedS1LY7877ezWr.jEtd5.nd3eYGxgf0bvur41d4xcd22Ct90cyesEFfUP4Y7..v29a+XW0pe4W2a4M+lNlW3KXeVHPdEuhiBuzC5fvl2TMyAVP.MiapCNOffx.KbshNPybZUG0v2W.rh2nFQc91geE8LLIkvITctoNZ08A.GYYgG3Q9fc5WoWZOHDBJpQaE9nlvN6X4G.JokCP4wQFEHuG7.9v6M.3yXgME.z1ZDs.TlZ7SiABWWoU4TO9UmKf6+o1uZOgc112.ZfWACuj9C+adH5EBd5H9YxEt+KAOWDM3L8qpA5hEbfFDyHsOUsr6EPilALF9x3PfCREwkepCcKHllke0AP.+L7b.Ns1Ye2u8Cm466LwZVyZ5nfmMU1vF1.9L+4eFb+228Uef1maNzH9uP5eBer7ZkfCMXAGp8eEZrfNNue.nF.q53q9rZlEqZLyJZ5jKlCU955VIFygZ9mruv3u.Th6YDoM9i0SY8uTC5sk3MfVsJHkVLAbFub8ulEOK3AaiO1znm0v2jZysaI.bKfNN7kUPqh8aB7.Ozel7KnGwPs+a4BBnST.Z7KkI..f25o+1le6Ln7s9VWNtpG5deWqHfVvxORNXta5Q9A+Lequ8kuhf4zUghYox+bLsQ557pCthRd9M5kFAmeNZqvujBOZJbE0QCa4yoiRAf6gC3jdQAouot.hN6xuFc4YY5yXibTX02D.MEIXFjgIRDqM4gz7nVlkpfymMbwwWnhwtAUZUuo.m5d8AqiGmQNDY9WGLp2a.5Y3kSybvg3.5OHCBxR3CnIOhN8WL4lBNafVSAecfNPWKTVL3sLSjJEh+BANPxud92YFEdy4zxBu2awTBiaNKWxjDnur1mx+L9O4S9TvG8i7QeVsy+kVZIbwW7EiOwm7Sf6+9tOW9DVhkl7Ow+bVVB6C.JP4NmlC5+Caxsl8INE+dFzJ186uZyvFAKwWvOvNS4oL.D9Rk5zw+U5axbHZfzEvLQ+nD3e0SeLND857VGxJtNqIli5eXn8ilrpYic3N+WCUgrKVZxtRhA3klItLe7F1y2Py13uQXkseZv29dx7A.vA9RdI3nO5E+s92V25Ciq7Jt5aeKWxkr3WxNqfxy7Y..0iD3Qteuj64M8F9odQK5wb33dkuR7hdQqB2y8b21.JcFKZFABS7WvPis75dNUeu.2VC6YVmi91BL3qETaa3uMXVqqEnvbJSBll4oyalTriniAejA7AUfUxzMAnGIao3YE.QvAMZpNDr09EwkeFAnsO8sVX8jikHs5fOC7kEzLMKETyL0LHISsrfzfeVAzqUXOvLG+M5m5TxzuaZk3+fQJg+YCFKE4kB2ADc1ZxeklE8iAcQs0GMK6tOWJd6Zormm8MYYf9MimgieNmoyCd1Hk8Id7iQiHNaVF+sJ0AO.1i8bOwuvO+u.N5i9nwylK288dO3u3O+OGabiaLLZ15mA4XCjbMIuoQi05Q+WqujpSn9R80u8jTcpUfmsv1PeCHcLEP6h3BrFLsObDZkwYjxempPAEHElNGoKQKmQi8U2+gaTPjTkQRVZTsKWQg3yTCv5x5ZwiYyV9c+epyvsqpOL1ikm8uEjBv3K8m73Up8HPyrR3yqzY+eYW9ki649eje1UDPqfxOxtZt9mt+64Lu7u6Utv0uTlg25aqIbZN2kvmERH6FVmWZYL30WguSS1jm0hHQcF29tep.zmyNocGQQbOaVcC.VuepaCNazxTCjIDm8U1glpTwQsxqUrHdc700Ri9sDFiwxuX8pO2jGhRGTDVDq4Y7ycPo2ABUCRsSGPwWQjxLuQrrrKvd2ATwszsLAFNI5mc7wYOQOUAJKDBGLklc04rJ9T7yFosM3jvx7D74YeSN78iHEwzo1JO6QscsnrzkJfwOmABFd8s3mUODnE823SHCS+g2heTPhif+nV6Zwu0G+25Y8N++5eiuN9O7696hMrwMF5+.GvOXYN.HYgOiU5BwRjn9WSVF7A2j+5KLqITG+6YlrNwDaVmVP70.Ep3ScG2z+CNCqzeI.OhloJvsS.31OzgeJ7RrAX5WC8spKMA0PhAhR2sIw3iKDWzns2Tv8qq+IN+Czttem5ueDbFToYW7aAwq1RI7qxA82M5pY+U2zylXKjUHBywHdBALVTbn1uAv9u+uH7pdUG+.dXb4Idhm.emuyU8fesK87tlEFnUX4GIY...3hu3uzkrpCZMO1q8TdM69NsSKFZNgS3DwW4h9x3G7.O.LknVnblNTQEvkjE+Xg2LgZF..fsgVLmsbGGzeiMBJvmtjqX6Nq8hF0HeaDpq2ptI4zKKGzlQeHR41+0N9HV.mMiSA9uG+YGowMVSbGq2uadsFAgsxunNzfSqZV.LiDZ.IMdjlosx+RiW02Zh0YeT78gPh9cGuZ+ew4CQ2fPcTtQMVmjzLYww1nAwERoDgKkTRNgGCOYBSCXj3EyAuEETLiFASJ4.NH7aNT3L5vCJXwW6KbfKgmI9pOGEfzx.zpyNsy6LdWuq2Edsm5qMW6mUU15V2J9q+q+qv0e8WuKyy8eQ2QTfdw9DMHzX.m880iBh0VNrBr6zd+tsWrwCRaP2D4jp1ddvb1NTmyLECe2.DdswgyTs+JMm49Ld308tDguz2.F+qTpECU8Y1xspxuRTlCosLpk1FhD95v2ug+79IyUcCW9PsH8yo6u.0WfKjDosTJM4pNVoGmBw9M9kjgJdKVeUsMNsS+zwyYEbcXeEWwUiG3g15u9BCvSgxOR25tGzK9fen8Y+1221K4.OfEp9ylMC6xNuy35ttqErPMl96lyMy4o9cfNMdyQX8i0K0FGdKxBcLTZFndyjV9AC+Iy.0aKmvD5TEY+yDsQi0JFMw3uXewcWTruGRDcwi+Wbfh7lvMYj+cbPOm7RPjV66skCvlcYwbpUoAZG12ve89BnX3u.Itg1z.1TzxyFN4tL3PNK+hiI8eyibKH+zAzgH82gvikA+jQYh9C1j41iBHvfO01YmUcvmwepNlwrQz+bv+pWyZvu9u1uNNxi7HwylKW20cc3rO6OM17V1R8AI4VHvQPA3k+sfyf7Q.sXMPNntZ0ay5mPjcd9atHri+mswyzVq3mxl1S708OdhC5ipCdmctP32AMlVbdvTnILZiDKEBdymrXsatAh6e.EV3Arhpr.sLq16KllrThur.uFw1H1eZwETfs7kKC3I9YvlCrgb9oufW3KDuu26YtvuOLVZokv4d9eoscN+E+Wd2KD.OEK+H8sywW5K84+C9VW125IWZ4V2lT4TN0SE669tu.vcbBjiDS2HSKiyeD7e4o0IzdrRo5rA9HI6y7RMvGcrzHK5xyeV6FBTWJ.QzMUhub.QexEO8XCX.ScpMfItJzk.OUXvSxOebK67QsQLPQ1CtNBe60GrZLstWGBt37yjLfcOAT2Pl9cD.um.r6tbGSPm4KuNbt7OFTFqiTR+chkMI4m5fkwQg+qBegge4wuha+nBN43o0FyKCBLdUbH1l7ZNvWRFhJt7ChXzOFP+5y8k7pfS+s81vu4G52D6+9u+3YqkssssgO6m8yh+n+S+Q3ge3G1dtI+TGN5y4wHshpu6N4U8OIJ+Fn+aAtADNm+7k4Ca+PweXbSKD0k3Y+Jsq6b6s6WxOZLNkJ8yOzF.432CDzM63zuuzGEsATmefpbP+SEKjtUqdRy9ii95ODx1P6e5E9yzfY+mc95AtUh9NB1uajeCugEQYoneJt+iKh38U1lMrn5D9HWd5Pus21aGOmELS3..W00bs3NuyM8wWX.dJV9Q9g2c+OvCtbfqZ+eCqZUunEp9ylMC61tsa35t1q0dlMCbAzrA0eDlmz4cMMVJ0QT58l8DbXpQ80mVeuoZyLkbVBgcylvq5vrojn2WA18DcwZUBdJfBs84fCJ57kI33YtaNBJw1pQPEK6.JS5KygQ+7LEM3ICZ4uVbiJUdqlMfRgOM.NMgYNG.z1.Tp3pkc.elpowcyICL7LeMXBY1wweNCRj0gNYcbF7vfOXfOieh1htAT5m2rcNNwf5Nx4d.dp6PyvvH7GLmV7My27paA02de+p+u7qhS7DNweh6s22JorgMrA7o+zeZbS2z5zNvfCJV90EHF2Pho.XN8c0KZ4gBAX6solMK0AKezM0IBfBriHXA0wOp6k1n0JNl3vNhybWxJGJgB31OD94Y3cMAVO0sFvvCGdCrT.FsmoerjTHUPlZNMs2ZnvCtpd6ihZzP7.Fk9swGMptgiblE7fAhAQvY5QsIpmRIWlDfvE.Z60dju7BjDr8m8a+1e7deeqfY+OMgy67+xa+y9e8OdksiAeJT9Q9H7K9B+r+1+ie8u4Rgcn5NnbRulS1dugaCzxB5VwVa7VoaVrn8y5lAD0MaCuwa7ljS+EoZHBMnN5v2+XBuzQPTemDnN+8iEHerQx7mkeCSotXv47rcD2LYUh4oYr.stryeqsJ7XFxSKhdShnmveokVxzfrVapY..P76H.vWavs52BZJ1H4MqGQnEZVHJ7A9WoAxUmJnFDnQ2fdBdW9CRlSyH2vCnFthK1.QT94OVwej9Svy0WDjC5cTVHX5mc3GrmCfS4TNU7w+3ebbvq4fwyVK5w66S9I+D3duu6ElN+Pm+v5mTWE8YAvg2fi6+3.AQu7WOQP0M+ZA46NfX2WIdI+PzjHsKlFQfNy+.uneIpP4vCZF24Y9STPu0OcIJXmpD7pSWiVCrGo+oMpuIJq7YKDGgOtvMH8W+no0+mnZVYVgunzewdlNieNHFo3xe6JXdXlFhxBOPCm+qRZMbM25M.va6m9suhtnrtgaXcXSa5N98VX.dZT9wx020K4kr5WvK9k7heMpS8cTY1rY349bet3ZtlqFZjY09sRqykLkY1z4eKUXOis5NA2Qn01PUHzO2h5mh9TilzyFA5fSwim0A585cG7tCCSwoMBwmYajes0sNMKUtcT3BzuNvHOKWh+IQpOqwh+a1eSdO7MGzL+j.nNTK11QrNYJPsK4urDH.Oh7XOpJ+b5Ro0jJQj90v5UZ034rC6kC9R72otZStFxhR+eUbEjqc8EtNYbNeI3GQ+Taxx+P1DR3bOddOO7u4ey+S3M+leyXQ2vt+jX4du26E+g+Q+g367c+NFeGRUeqX7uFPJbYNR8IF7I88v9DflbPUArz1nq9uw2reQXDqqH56ohYQDa40DKpEhexQwYLRj9y.v9wiKmnQfI3iRPG9r7aLdh6y.lvkPS2Rdec+MzKTBjmImHBPsinA7yAdvAJ4YIPuWS7M9mCOgOMJGp8s4QPskFJf9r8+E8hvuv688EteT1QkK3B9xK8W7Y9O85WX.dZT9wRN9tfy6y8A+5+ieqEOE..3UehmDdQqZUHa9GB04TbGpcixoRAjyZ8d6exi5ru1MTwCZofKRiEnOjH.S4ssIVJif2b66OePzppiO2jEgtVvGIv8AuF867UO2Vgj4hL+OR9FGPGYa97KKnXeleckxiKziHXw3+DBM4WHhmN529VvnKunHrjJXUeYfer7KCu1p80yq6ncVLiem5VF9mne2VXxn0xI+Zk0t10h+s+a++7eQb799c9c9cvF23FMA3Nt+aN8+RtdvG+HnSNSMf4TDvm0ui89w+7K7mBZY6tiNgEbYEKQ5JnlPrT23WyNw.9BgXjWd7medh8GAu5XuCd8YgnTLCHXTY33OwkKZnF4wOFela2Yo8Bv.bpKiKD5cjg9ajuAtElDAu82w6bEsy+W2McS3V1z5+Ouv.7zr7isKv6m+K9.2+U+hewm3BmEfRAO+8YevUdEeW6Y1rrSBbaxegYTNHhKME9.g0iwurd31jwSTUR2SBzSlONgG8Ne7VzrPHvwQ9tOf4sLe363eelzZlBLYAxzu2lzvgAYZXL7CythRiA7KQ52hWSa+BJypyJZRf0ALixHiwiJ+2LlwmpBWVw8+hSTXNwsPyRmkDcM5.3U7XOmwO2l7Lxo.V0YVkumA35x8enw6gOGvebO.zwSomo0am2kcAmwO2Yf2065cgccW20rD5YMkst0sh+z+r+T7e6RuTLIScovuJxc4mIuowZb+RtOQgOGx.K+YIa8x7gCP0gWwmTp3uP1EDQ6mDnmrOOPxoVv7oEnLE+hyiB4t0weT+DlSSKlkvyH8+A0a9vKQ1md1nFPZxCcSQNAw14+7.O6D6nQozD.bl.z8oPby30rdowT.w1RAPPWlF79zjDMDyjGLPf+gpqU4kC5fVMd2u62yb8KjKhLgy6B9xS+U+w+QulEBfmAJ+XKee+segO+u9gulW1u5QdDG9rRYwhH53NtWIdYurWFV+5We6Iwn8JjVVH.x4T3tg5sKkSG7KtGD5L8H7JTETGdBhAAOzDL8RkPem.no1yi7jlIWSAWZiN71uPh.ePtSxBCN7MgH47NAeT3jDfBr6E+Bg6NqII3sVUlvDlQYAv+UYplBMdbNGEtdRAlIndGBzb5Mij+HOvJOvUha5pZC6CbWN3YC+gMLIVF3SEAn+F6ilEUGzClAu5rZD7dpulC9yAKzp8pW8pwuzuz6+Y06vef5w66u7u7u.Oxi9n0Gjxfh5LmUVs.ZGMK9Q5Ol1eT2NjcNNZvVYFP8JGG99...9Y9WjN0md8AeKDVRbRLagLgMhcJDoJA3YSKIjq+e5mKnTT3GH.5DowFn.zN1u9oh.ns1+KidLyAVZ2M6WRHvkL50VV+E8ZE1tAB0KuMClQNQZcZTaFShDggJSB.felel2EVTec..W+MdSX8aXC+GWX.dFn7i0WgWq5.dI64AbfG3o7hVAFddQqZU3acYeS+Aly335n1uN4sJOnTJE6RzPgi2U5w8Z.a+0chxq4rgeg9bDg0WpDzL0LsVKHhhoboFsCYFvYM+CjOHqNfi9tR34cIN.uq5Ib3hXBdfDqFjCgpFvuGkPo3uMAmw7.H6lyn0IsQK1aHsRIbbA0SVgdqJZ7uK9Fx+wtj356OW9GvpCKHHssP6EfSqSm7mDXY7Kb++xCOGDQG9EDZ65emgS+zea3LOyeQrm64dhmsV1111FN2y8bw4eAmOdxm3IhwiF5mrAGTPAs+CqvVqjqK5Bd6AifWWme+p7EgM4WwqZK3jpib8TxvydUWueiugMhP...H.jDQAQ0uhHv2reY7CXJHAkhQqGuZiQcf4Jv7RIpz3P3C32IfvRQxsG0.lMFhAz8CkDXp1RylCrg4Lp9V1.LoGrf.JTamRFSXLihu3r26wqS+RPGIt1+Q3OxWwQh2963c10dyqrzzDNuy6BW5u7O6+7OVuws9w5474Kc9e9O7+v+vWeoUx8BvgdnubbLG6wYee35HocFAWK8qADfaL02s5MEbQeWbadnbELgbpqc3ll.BJiCmTHslS7aEPEQy.70kRS8MbdJpDGQfRASA7WZvQAX3B.GtjwLu0Y3w.3iJ9VSkLHolSz6fAl9IpHd4aT7qCUsOIbABYsO2+WrYakGrCpuwwKnQ8vblF98L7c3OVr80.kwAW+Q5j+NRh3eggOYrhwuxHL7nTvAevGL9o+o+oeV8qt2MrwMfe2e2+83q+0+GA.4udGv+bLgi5DsSojN7mgGI3UDCPNyb8Wdu9XW60kn9OYDwb9qa58YMZToo7d4IAdv4aH02pMwTfhlWOBde7+.3QV+SaCDpaNlDeCXJrAjg66J8F3KbIHDpQIz3zhp.M0+czO7Mpm9jg1O.vHm+VaazOsvJQ2Gokl.3m4+gU102+0++O68lG+cUUkmne22jPlHSP.xDggDBHyNffVU4XMQMzkkZqkVNzcUkRo3.SutdSc2uW2uWUcMW97gsUWpfBwvff.IfLJffVHigDx.ABXz.HiQlIPxc2+wYuVquq0Yet+9c+YTBP17g76dO28ZuF168ZZObV8Zv8e+a5+1PAzNfxuz0DrOycA61bm6971Fs2K...KXA6K9dW+017Ew9ShVGXXCPbt.LfTzxYAvsixS1fHyXJAm78jevV7DAjg+2aZ7lAJZPJojlYIGop7nhHkQ0.Gi3Tqj4.Sha3fy4t0QO.uKhemEUFdh1BNC3vORjANYkLEYjsSF5QJMkHkbuVcI7JsqrDJY.20OLK1SjBqrS.5a6Z6e.F91NS1deHXhYSf3uc9rOax1ba7SC.L8UNss5uwPn7BZ2+kAvV1xVvTlxjwALDuKx2YozueeboW5khy7LNC7LOyyzV1DhnmkKI3UvyxF0QXtyDztemf2EoOnd4hCp509Mj6ZD618S1uKRq27ukw4Y6ztw61eO9gui0oZI65+EYh2Xn0Hh3JYL.MSrC3YYSQATDdpJDMYi4E4l56cRjmk8+iHfpXK1VacdZRY9izdN.8Kpf7hVR9rnyHmEi2caqPPZl3sFzKxrrSVbLu4iEui2w6ZvsGU1d+937+1W71Nqu1W9cNpAZGT4W52zGK+aeN+edMW+M7Rae6aeTCybm6bwu1a6sqeWNdalwBRAb173z7RLLhpXPRiLsbCAp2M2Tj+xYnMZK2z7GoVitpU50Ko2GAMQ1pDE.jaIPg9QgO8zuNvq0DkhgprstWxYHlUZTaiMxvmqAOhv2BT6u5DRg96q+HezdMypIWyzmUnV9e4VTCPhvxRkHWzMgUg.z9OFKEiEUzyPJJI3K2herCM0keEXDEUfk+Iy2jhgm5obTF+VnQFdXv6nemyAdm.D3Ed+htnKBO3C9f0n7cZKOxi7H3u8u8uAKeEK2bVe.8exvcq+iUvmbxFWgly0d9SRG+wms+XIQC+DcI7tNuWBZq6LPpzunuJo7hRl7eo4ztQ2B.IatbK1TL70F8c.uWOmlDARlYMkPvzX7x20L6I5QD.GvYuW0GkDLH6IJVvfxbqjIaPRYklW4xFCzCI+sKXpt1ZU2KEXgzmoNN.35GG+3m.dOum2am7Ssxsc6qD2+l9Q+uOT.sCp7xRt.Wv9t+O5zm1z+8128cAiZX1+8+.vMbCWO11KsM.HFAEOwjZ0r2.bqcOpavRLhJmWcTNu9IwQ.1nTn8I2ba7AfSOLkBxZ3UFOI64.PzWNIKQ.0dFlLhIa7eV3EwFiBjsNxr4UuLw7jMOLv2A9Yi+Q5W+4xDmd8rK8DqsR5lBjuNfkMNj9NYRnOJhcQsoMQO0B+NYIEwHuF6VFdRVkcYZw066kkIONi9DwFucQrFZe6uIG7nB7l72i+t3094L138tQ7Vequ0WQbS+ci23MhuzW5KgG6webubi4eD5eotP4otsCHMXT5+rs6mU3dE6D6P1CB3OCK60IXiWKpWTrHqntDwOmtemdGVP3HtJ7uN.N.eBAdsF+S7TU36.WiF7SxOgrkJy63+tt.dTZnkCuTFG.4NNM.vM+WqLsbCj7uRzTTegeYDah3unaJ6Wlgeqi+3wQ+5eCU4kZkW5kdIrry8a87myW+q76NpAZGX4kEG.V+ZW8sNq8Ztm1wbLuwcazdwiLwINQLgwOAr10tF.HctRGcyn7XjpnXvt0NdW90hQtdolnJkcldhaWS2N3biKoPmoCBqNb3MYzTZVqa9v1TTPTPobb2pheh+c6NeB+ppMkjIqmIP3Mo+wlCUC9QI9qY0SUtX2fYrwdNiF0b7Rb3fO0Dx8pPhpnjQES9Izbw3nwhN37HxS5R1hRc.Oy5sLRKxOV4JYzoJ9UBfnEh9E7LP7Gnk3oI3oe5mBaaaaam5WxOO8S+z3q7U+J3ptpqBae6aC5jMWTe7rGvVlpJK39OAdoEX4L.zM3GfeLmK8+IJpVfVuWK38xhPAxl.TYGpW0QaLQ2lvo1iAgubhrpYqOepC9OhzQI7UvuPcb2TSD3PCzfgNarenPOPiz1qbokZGyCAeRwDmDR1RtX5wZaWnAFyVhjh+TMZkrWLiYLC7I9DmvPcm++8twe.tse3s+GsgMrl0MpAZGX4ksP.djG3m79twu++5PAy63c9NaN5RIKRR.KMUtaFOvoyt9FBTfqe4plrIhHw3eo8YGDI+Kxn4ErRlFzYnOyZkPqYv5K3h9tMiXJYu0q5QvnSbx1fwrP+EOqE7KzlKBc8SlRhhjyfOORvGvetM9qxtpW+.HWdidWdSB1rKbaVEecmQiFGgDbvuWAjkMQWi09xyZdtlUGPqCaHZQjCo20FLXNh.RIl77H7Yfb+9LaZJnxFSmg7wbSanFVxF9oztZF+ImEnO6VdFF+jySBLJ8RetQ9mvUcUWI1vF1.1YrrpUsJ7e4+x+2MuOPDAnH+..JFhx8665W3r1H7rZGijeh0Bq+OSY0potbF5jMhZ+B7YBdMKd4bYC.VbdnWRG+1PVMXhey1pi+ImBECYNiMry.HP+xzODF+pi4XC2x3r5vCMPJaL2.gGsweRgWnD65VuOxMQ8KySKWHasWBfrRG4ByX8SYoKhhpW9r18Q3OqeNkgaCIqKuQqn+qXKIa573n+Q4Y8QF+gu22G1sg3N03YetmC2vM9Cd7ku7y8BG0.sCt7xlC.qXEWzU98tga7gd5m9YF0vLtwMd79+.ePZGuaFqaJYx3KswVx0y.f2sNTLH2LkVN9Y5ftDYDMKNl6cITLRmDOTz4PpZnBhJC3B63U1Nfp5Rctw3otNwAYh1LiH9m4fGiF3y1yDmGLAPkRXxj37jIRfsiqMGtDXaTJSqQIncnrnjsK9OCcYAXm.zcHegm5j+EA.H6NB+VAdNiHEp24PgI+xlRrjcbuZI9x4V3hgef3OY325qM5W9rz+kAvYdlmAd9m+4iTwKakW7EeQrzkd13z+RmNdpm5oKxLRoOL5G.pAKsDjepgesyTNRsib+Oxlil4Lz8jh.ud0dS8+oLzW3X1xCjI7k8uuLBmQbwfpNmQaXkwaZE0wv1LfNGqHeDm7UvoFOU9mDCu37MxiN3ageQWYC96KyoSnbyqZxOU2maW+q8xpNO1nup+xrOq+aSc3TxyqUeYYH5kbq8eKV.lLVZON8+ZfgfbXD.K5.VDdyG6w1p8FT45u9a.OwC+3G+PAzN3xKqKB3C7idh200bsW+PAyQbDGINhCu4ZKU5HrnnK9PKdqAemWKiSEKtRczMnW+9zFPSZWtMjgcYtYnlUhJwpSU72Wn+99qhREu0Rkk0hlA0.VD5I6LI47XVvSmvCA9H9oVfmjn0hzdUwAAtdhi.auDIGSWMNaHmYflRex.uUuFGJjqQXkuxwykMsAKivKQRSvq7NW+nLg3e0fC5P9IFWpR+Q4btJ7Q5Q9b+JsatB8G4om3IdB7MOmkgcFJ2+8e+3+5+O+Ww26688fkYoVSr7J76T9YOSqeE4mH+2dY7Fm.GYS9kxVqvim3i9216W1g+D7MyMyTvso1A6JGyuZFTGAGrYGgL9miLUUB1BVV+APYS2VEdLxvmZieP32doqIPRBg3eKsglYBp6OwvW9n2wDqM1dNqYa.rq1kN.mS1UJ01btxyi54kxG3C9Ggg4R+4wehsfevMdSabEq37tkQt1+hq7x5ABdSaZcO1tOqY+6c3G5qadSaZ69nFtC3.OPbCeuuG5matHGZcE9V5zaNpf7uAzwrJMM6oTprVQ1FBTGFIo3pABxARZsx000UbGIRaTQxEHnMEnrVyJMzvO5EhSwAm1qoeYuNDxJhakqSIBjjEA4n.di+goqzAO0t7uYhH6uDLJNREGBb32NpThLBnjFUQFky5ZwJ3OYc.M7rnDpzlt8G.+WDz6JYHJ6kehmWQYaY.j1PZfahLqPOpJoPaavmFcvKQJUHW+3ujqc4eiw+Ct4Mi4Lm4h4Mu4gWNJMGuuUfy3L9Z3Ydlmg68ZJ53HaPSj+0O4j+91IysoTOIq.Bs3vqgeAK8H3yE3kkqx0Up3L4NdeHIeWOLrNGOUhLNuQ5+.UOkGsMun+zNz.rnNLCJKnhnrz1Mv6irVkv7.PFd52L7GEflNR6R6pfmLZcc+F5kJ3f6+kZIWpR7dJP1qNldWAecc8B2tTISwJKy8yDSCfe021aCuM5TpMZJW7EubrpUdyG7l1zldtgBvcvkW12FvW38ug21kc4W0.6Vhk8Zu1a7ac7Gu6kJCEWL4gJk9MZBZNNJfRKXe0M8rFIYOjzyYdVCkv6gpNnlFrpQxj.jMinPAMHyrLIYwvhbkoeA+AZmhvWWa4RcxB8wy1yNxqC3yd3ylyTsyHfGdmzU+GUn2R4lW9Q4.I2u3rNsj.kI38IcT49VDVN7mk+lsMFXYhKeMf1Om8KOfPwRWoH+IM7N9WwEiesSn83OodDLJTsfO2Fd46z33VAyPQ46gWT7RyYJ0coK8rwV1xVvurKOxi9H3u9u4uFKe4K2ewbIQuagRCIczN5OJ+Uier7q4IhARVJ2OaqguFMbA9Tn+Sh32Vy+33OCOMioZfseYIAZdU9x64G+XgV+UmLWl+AdYwX8KTaAZNNH5mz+3yHnE4tEUrgCFdDgWFOpzhQ+8ivC4J3sbLcsN.p2fE.JEPyqB6oCDwuQGZFvJ8C8PSlcGwian9WpMU1k1SBv5dl5TmJdOum+vNa2ZkMde2Ots6XkWx0e8W+iMT.9KfxK+WIXadyaaVyddyY9yetuo8YuGcunf..NfC3.wscK2Bd9m+4nfnDKLVj+75AoAmF8xCF7x.rTpmsS7A7+qCO9mKyCc3WaeESdDWpPuTO6kSglIfle2EOCAt+JHFTPjd7mnu3oo.77yinLFkaPV5uxh0Pnr+TY9GGvBebr50qouPdQoHmeZoFRS0KQ6MYsy1e5Ez+VxFPl3S6BDpA9lnyrHWX30uI+XNR+Uje7SccFNoiLZRU9zM7swuntTMxm7R+b3ywmssssMr4GXy33N1iqcVp9ET4Ftwa.eouzWBOwi+3soe.mfUmSFoeQTPSvhoSmFR3iBn.F81.oIKbYKRUAmxkMEJYbxAeBdmRCN7IzjZfFYm7OL.qZoccsVv8oDrwOHBD8rxijwbjUt1vSxrtg2dt5vQkQfIfNOtepzQzU6X7lujzmw6Cn.8Wb3Wx1G.JW1aTSVaLd1zE5z0BimDBvNMCY7g9v+w3.WzhqxS0JaueebNm6Ez+L+W9R6TbDbdYOC...K+BW5m5Jupq8k11111nFlILgIfO3G5Cqd44FcJCDQyeSx.ay4xPDa1yjloe+95KNB457j2y2bjWr2mxfeYPSyeRv7HVagV3WNepRzFxMHVJfeahPCRcdgKC7U762LXl.nC3gm9AwaNYlFwr.uASRwEk0Cmqy1eYupEi+PbDR5GRPuwFgBdym4ShgbLrxIeehsYtL9W2rfTTJHm0qBZd+QHhgLI6bq+ZNJ+DEcF+m0J3kGtwOrhxba3Y9uN9MKIL8KHpZ+WA+qe8qGWy28ZvunKO8S+z3zO8SGm023afW7Eewv7GO8KioT4eW7ePl562rww8Cv2y0tI8YrAF0PSN61OIwHNKO0tnqjq1Z857EfnTZ9G7yIHCMxn7jhnxy44LYD9Bb7sM2z.W3qjPSD7oH7YB97HAeisUV+U+B96yW7Ngq5WtO18LxgDSNmaAaljkpuwx7a4dFPdFqvQASzwF5eSFcTZ.UOPNmwhV7Agi6s7VvvTtka6NvFtu68kkK8mZkW9y.PorO66AdeScxS58t+62BG0vr268diG5gdH7POzCB0uMN5zhwvTpsWcbj6bggWilpDQUOe3F9nkRsgWijfCGg73uVlHj041plsWD5E+EG5E9uEGQQylp9b62n8.fQlPtODjpa9Yv94WTVEjIVqoDQ2+Um.yZGQIyLI6sJnVO6YZfJ8r07mk+YPuIFI5uGUG8jeDTPzh+09OhAj0iMQaaQhk45x323+F7ZGYutEYY22C8WUnea..I.qf+Mb22MN5i9nwzm9zwuHJqd0qB+Segu.9w+3M0B+N5OP6xT035AWk+gIe31vpIYuLzu51C.vviMNO4fW9tLFTd4hYA4laQibyZzGL9hRCWq94P2GOsRLR6F+Ef0F+jceWgG17XgnbvmFL7MqyusGaRD74bywuqhMbpsXug.geodIssHsiVsYi0HoAQfRl+5L6VEzlgrG.XOJXJx1SE8F+3vm9D+LX5SeF0ayJkm+EdArry47dly6rNiWVtzepU1owAf6YsqZ0Sely9DN5i9Hm1jlzjF0vcPGzAgev2+6iW5kdIWJDS8T0+VkICgiTpNkeuWpWytD2oqL4xlnkteKhsdZ6yCezFHLPiGDCchLqZKAn2HdZJ9I83lxe.1PtmPo+zI7lQWS9ETEknFiTZw7ezQB9ytaNPtPU1De8JQQkMm2SohLrrFrrw2rfilFrYITD4WS62KAcCDZKySCejKcBwTvGqWMc1wMLoqS0I.LYsSLvOqF7CB+rsCG8WQ9GwU4449Ybua7dwuxuxuxNzaIvst0shy4bNGb9m+4gst0sVG+A5GA9mcIrY3evQorIHrivaCWaWOzk9GY7S.+xxMI0QVD.wfStPA70hifV47k2uetDQI8l7icHqlZmTgQX8W7uGnS0Xb1nFh8ICYJ3.J0Ci5o5psCHjTnDNx6H7xSjK6ml.m8Stsrgzl0cYZTze3pmQmJA.nB.K4E9LklH4dWK6.CWisduyEpCQYP8gMO83O9iGGywLbG6uK+JtZb6q5ldma7tumexPA3u.K6z3...vVW37W1T151N0i3vOrQMLSbhSBSaZSGq5NWYyCZFcRFC8Fnauq7qYIhUvzLwtZD1ZDDlk0jzZjQNYG6aqpHi+5NhjREkILbIJZC.hlXi0kO4r6avSlsawSI9eczu8b46d3aKaLmw7Yff6W5f001RTvXFH6AjRph5jnHTM8ZTZJo6kSuyPN5OvyjLRclH1mVg98xOiWyLEUoRIzFtNgmwOfyHYsgQ98iggGe+ea53odpmBuz1dIbnG5gVgpF9x8e+2O9m9m9mravyB1Zg+nPHXLLRqAaDtO3Tbq.pcrM6CDxfm1lNYUROdb182uwAVJpswpgdKyNJ++sn4.aWXfFUS4VUOEZ7TKfMpz42Xg.T4elAIEAG.wLYUA9jutQ300deD1889.J.LGmZyZtrPvPXpqzazUYuYLvf8RLlCSry95IDv9Lm4f+j+zOwP8V07AdvGBm+EdQ2xJ9Veq+SiZf9kPYmJG.d769te5oOq4bX6+9svCa164dLpgaeW3Bw8du2Cd7G6Qg2PmYfhGvvQP2oS.TTV8JFiiG2PwSyF86IaRoLftz.wMph.uNQm+gRQm3n3kSGVpbrAMO881kHCCQ7K0HIFXH3YGVnPaaGvt4fgEnk2vnnt.HWMh+TqF.0UR5veIRLIZgLYhLYQjIsos7.I8nCZh6lF0Tbj76wBjn8fgkYAM9fDugwDCtdKt1Hqn7mD3k+1Up+UiYAwhv+oP640KasMGAbm3unwci228gkrjkfYO6YiwZQd688U+peU7rO6yXzuvjA7Wi94+ZvazOK+gz+UZfbAvjrLdYyfeFz6i9hgVMiPzPR+es0GWtcJ62WFG1GxsaIKkcpVpwXgeT6+IdhclgDfkeKqORbXv0KmrercDzjSPT2fMiR.WabO8krqKYN6I8QbNA75FH91uOihQemM5Ov+pnTXobydu.4rKSLIHNpU2AfXT8rCG79xIAi96iL9T+4mH1q8Zzug0y493adtWP9q9e+Kbf.Xz+Vv6WBkcJ1Dfb4h+Vm0G36b4W4Psg.A.9i+HeTL9IrajArJch.dEsP9PEEzE3Sv1HI.Y5s2Gbq0rrou3wZ5.bZ1uZiKKTVySqso.0KJHEddoKBd.qQKk7OKwSYRjA2r6usS8HsQX3myJCTn4cusUW2wogLpoJV7hlZrUa7KehRKYyltpe4yIsYxEEh8EEDD7IZPPiwfB7HoWoqtM.UAd9rfajTpbYxD3ekE800UJJHqJ+H3aa6X.8+hLfUrFspEvutA4J+9YblmAdtmarcDkejG4QveyeyeMt3K9hQVbJSn+ntXRItR+AVzwpVHf9lQ1rcIqORy.E0.xu4Nteojll3dIF+YZbTrGvF+YKSUXnreRRKvrG36+0g3x7GwXXh3ewYmBL8qH+TObbNp5oI09rCd+3GNb+H78CvyuLyzM82.RAu3ohsS6C5uJ5F4.kLs0MBybRtLmRtktJtQNqfXsMi0wB5x+aui296.K9fNnt4mJka9VuCrg69d9OAfWbn.7WBkcpx.fT1q8cdqaRSXxefC7.1+QMLScpSE61Dl.VWIUi.MchtArrw6D4gJPmdIlHENhmt5lbIF8J23xinLIj3JVvOFA72r7CHrb.kA6tH6ISjpQSaCPxQ2KNDv5RquWILdQoyRao7u2tQj.HEJE0Ys9sbKXLDQeOvlMzVIpqby6wA40IdOxHp79E.TynJ8RlwA9HBJqdqbYCgDrHHKvHW9P8o55EfYsu2gb.RgrIW7aduJvyFny1uI9wDR.i84lzcQtZRhUxYt3zjm64edrkmXK3M7FF8uYy..tga3Fvo+kNc7XO5iRNczP+r8PG+GnekeE9u.uLuieg7zuzfR+GuU1z9Kjn5E5KK3WuvdTYmIMEi.hCAMi4fM9CYmL2ANzlQY73R63SNiYzypgI+byk0ufJ32qvKGZa0LM8bmqpECyRG.GcdeEdaYRE3k4C55t6R+uXPljIpdAT7ow7LwsKnHiv1GSltHXW3O4LzH+qoWUOsRI561uR3fU1.LyYNSbBepSDSXBSnUa1U4YetmCK8bNum7bO6u1uynFneIV1ozAf6YsqcsSYl64G+nN7CeVSYJSYTC29c.G.V+5WGd7s7DpwIYnoa8+0YKlkhQ6lBLkj2geY54TZlHii77ZcS8INkj8FX8Q16m3ly14hu89H.gM8jXbOtrGxjIZCHRFXrOY7kyGYkN4eKJ+n0BLnQWbDw4QRxThj3mGl3wEsZB1Ja1JdOazX6nmtFfRZHSn3bPRQOgKyPOf+00riTZD.pS.weWT.xiQz9GVFDfSjurYG1TjeeLXNAvjEfvrvK+ozfJidB9SQFZLv27le.L24NGL+4OeLRkm9oeZ7U9JeEb4Wwki9ae6JMx3wFAVA+UoeR9En09w1nTAyLURuPsbi0alb1Lmp.s5PX4akYzTFHrcvdy7PgC7txvxuvzTGwyysb7eJvSEZW3GWcIu.b+NI+banNBXcIEbzHS.AGBrI05b1n7WbzJ2mvIKOHbjRvCMqxKyfxskeeFH6ACqs7Y10cmRDJt8gjIVJsK+TI6UMs+m7D9TXtC4Ml4xW92Aq7V9Au4669tuGYn.7WRkcJc...34e5m7L6saS8u3MbzGQZzd4jzKkvRVxRv2+FtAze68cF8jhaG6GRa1HlE.zDAQylLoG861G3cerE.jYxhWWbwoZ+lnqNMX00uA.Ar2bYb5eSItNdCEQCTBsjRBQXFq75wLqus1w+Y6mc0kvoiRBYOPoIWTizeolUjCh4wF8gEC28LdwYyTkVhhMRIbwVij9eQgFmI.NR+bAHIRR0XWhuRhQ3eqw.j.HHeh1NxgO6OEAjvhF+AwPQDWC3YR2uzDqcsqEG6wdbXxSdxnqxpW8pw+v+3+.9w+3M4lqw7ui9i3uJ86gWirDFOkH9OixqbVxPQS+Hz5Z32cKbWJxtg2bwVGio5JryAtxSwwnwouJgXHiMjEB+OnqxKcrgMdCzItYhikH72ptQdQA2OJLEXVYG+av2H+ynjws51cgEceaC1ZVIH4WJBaot8K57xYaOGzmb7nCzq+hEiAu7TjtrVTPB+Zus2NdWu62cmsbsxFt2Mhu8Eshk+ctzK9KLT.9KwxNsN.rksrksN68YNoYNiY8NVv7G8dcM0oNULooLEbWqdU.vLzqGcOPcqoDRgYrcYDNkRNsFoDTiExfFaoAnoyptU6yxLSyHkeTeayAMkd8rnREiTIjbuSscpbSlxwD73WMbwV5E6AvuLANmCHaFpzKwDPTADYboPclwdVlyh.+FTrZgjSD1MkKZl.rMvonpRfRxNovVdkrsoe8ULLn2EAk5X1MMCuY6ijyU4VNq0xXHokN1m4ryH3mjqx2bQKE6KT736.XEmBLaaaaC+jM+Sva4s7VZ4frb79Nuy6bwVewWjDZnxm8Q96vOr4Ob+fI+L4ukx6j5vEaLIy3rvaxtBOSFmj4drLSuVeK3TW47bYlsdz9roLtgnwAjkO2rWhRFc5plUYN4WtKkF4Sb2WKDYFX0gON7KyqrhvAiL...B.IQTPTIPSAZfwolgN1gijhkbAL9t0.8yA5ynLTvuBnIc8C4ENgZGMPMgLageYCAq.TiHPT9G6npgS.f8bO2SbBe5SDie7iui1sc4EewWDm8ROuW5rOyu7NliSyufJ6z5...vcu1655lxz2yO6a3nOxoLwg38r79ue6GtuMtQ7HO1ipFnyTXiwcEe6Hvc9na0q7KiSLDqJVZ9AdCDo3fM7KS2nnmaTxWbRomOWAQ5PxBftQ2J3WiFMYSRYqXrAc.+kUio.n3jjf0JzuZXl0bxTHMC04rEaQM6pl626B+s5DhcH0Z2hx5LR5tBVu1VSMbh89Ao468J7Z+Lz2m6ZVAx1NEWy3Rwfi5XPIUCbcyI5XKhFEK5K0Hh74kIv7Xvj+bT6wSL.+yNQRvwAK5XSQtWLZJyY3e7G6wvjlzjwhVzhTQ+8e+2O9G+G+GwZj8bCsLC5mozTKDnbUKKNBw3R26FD7tyrexS+rSBrbQk+k1VyvUg10crdNq3TFKHiaaxjj8Zqt1Nlu03xV1cDp0V2ccbZX4Qzt7.uH+tTA1w3ruBPVdOqsX7WQ1FnGIf.aNtLnxNwQR+j.u1+0mtneZULZJSsC69ky4Gvzuoetg7kc7eSakK6MJo+ssVaoMkcyeV4Qt+h2nxhtPA2mvm9Dw9r26SGsb8xUdUWCt8a9G96rgMr9MNT.9K4xNcmBfX4Q17Fe8W5kc4CELoTO7Q93+6vTlby9GPOtThG9tvfsRNyd61pQ01QdgA0riS451LkPFnix.NNkSN5QgMaJEzeirrEJ86Wdej2qGsN0I55tMYyhxzf+Rax63et0yvlHwZ5hSdsvnBDVi.zPu1llrP9CaPpS7q3JyhpVkTK3SPiZK2We+JnNADNw.pbMaJXRYwggrtglTiPE9OG3e4UHMWW80Jsv2E3k2m7Q4m+kSjH+IqND9icflrFTfVYx2Ka4RpMJm2w9Q3ufK7BvC7.O.52uOV9xWN9K+q9KwC+HObPIJHZDpgClm3WpNQ9WF+JQpyxOQlxvKui4Y7KWGzxe0WpTvF2oKWTJ4VFf9YarQyQ7qWQlz2bYhM9Gs3nV3kuKoF2bdnAdyvrW96pj1bYoyHa6Jcs+I1CRy+ph+v7+n5KaIArwb8y4v0hdCM1KP.88qohPQFMUlaZFak4+1fNQ+mRqv5u5CZoSgk8Az2xlPW63+DL5VcFiMJP7OKid2+5+5XIGzRpzlcW17l2Ltta36eKqXEWzUNT.9xPYm5L...rwMtwmZFSe166bl2beC6y9r2iZ3lxjmL1y8XOwse62FrcCpo1KdN3kmYeFvzv5qirjBtKCDBd+ZzCz0xO3MwKzngSClNbHoeYsvJYBvcDlRMvIdtGwukMDSmSqMRHD84RTnVCnNTPFlhQPRUFVTH1yTvkZ1JBXqpV5pYgn+uAwmu2Kwl.a3IcoADGRRtdsx+J8E1cIffL4LlKJl0DijgKy.B9SvTTwxstfOGjk9cAuwc5Yu1M9C9Oy3Sk+9A.RTW0fOmy3d1vFvMcS2Dtoa5es4GyF9E4m.iR25XivyI7qmmeJCPY.2RtTCdyHewfXOaLqI6XIXibVxBT+98KYGp3PXOYrPVqOOVpkymoJOCgcvOCuihH8GpCcVsrOXMf9byZGTiUIatEs3EC.+dGRbyWoffxY5p6m7sAHoNVW+l1yLpqMKKrRz2y9J5oe9495Ju6T5JdjFvys+AGeWYyBV99BVvBwe5m3SNT2Jlae6aGKcYme+u1+7Wb+QkQG6rU1o2A..f0u965Rlxz2yS80ezG4Dm3tsaiZ3l+7mOd7G+wvl2byMunynNMgo42hNHH00O5gSMDfLAI4fSlfmJFUkMHirmCDuaMcwQ5vZK627Nhfrsm.L7ao9TdaXwq7ptS8yF8oF+Ak5uhFiDIWbFHyvKKIqshMbqATzqvXa3GNMmL9gZLvy8Bdbdh3mlkr+5g2LxHo+qWOKBP61EzZ.0wLX0A.ji.laMxUNrP2L9YiXMFLo8PPta36wvGTDyGITuwZuSLQC4V5bKcH1..6MgXGv+LO8SisrkeF4ffM1fc1Q9NTdrPzIYeTjc7erOlg2L7ZNujy1KsG.a2fG7az+FjjFyJQPKKMjbCSlylqUZy3UKXOiG2o.j6DdWvFhHWD+EHbF9feNVFLZnFP0k45J8sw.vOxFEyQlavKGgRabpj1cI3i5kl1TwelnSY9uPKk5qWKuNIn0mKONhekZqD8Oe5rrhpg14TBeoAM9Ira3ycxmLlwPbW+C.bMW60i+0a8N9n2yZW0cNT.9xTYm9k.PJq9Atuid4K+xFZ39f+QeXL68ZuoIa1Ze0nHnY.faSjfRTDzuaOu8fr9fRcHfpvSN6vxZygP6IaTsx2fPc7EXAuYfhNTJKEQCdKlqTbiRpRsY6bzMFOa3RlPZOS3Wh9M2v8ohLysqw+R9mE4Ouxm5uyx0L2+XOWmr5j+wuCSYsBeP5IqqKJ8YYJkukFnOQ+5tmHUjmYS0T+L89fmTD1Om638IeIMlJqxuO587UDdE+JS1zVjXSgsAO9eWLPKzlR+BLD+OP3I5ueXrJfL9OqoyuYoP.kNehKc7Oy6F74.77k5C+lfTwelug5D5KoY7oY4HJ0seeZrtRIVFDjp5sEU9Q2GHFHBuX3yRGOy+J9gSvzR+jK0+L7A9GYipbyeRL944ug4ePzkj88uEK4N4emuZeyf0+I3Wmtxxm.9Y8t7umRnz+a3w8FBshdYPzugCqCMKcTBMmrwL+a+.e.L24L2p7WWkMu4Miq5Ztta+R+Vm8RGJ.eYr7JhL...7f2y8rkIO6YuWyY168adtyYzugLF+3GONf8+.v+5O3G.IxOep94OCMRA241t4WQsJpQRkLOJSHANRRoJZansaJzrx9LfWN.zFdtTte.P4nI1ueeZIO3Ktn.ex7elddY9gEINQewvgR1yxZaIJbiTahACPZotvekPu3rq3.JJVpHlrnOh3ump7PSGbRxdSodYAVqGVZG8MTHr9OUtpRsLcgBAM89.TjxVMIdzfGfLhpQN2PmM8uxQipAZ89WQbFRj8D9SYd7wXCdKBeTbxzmNeg2bQrC+oonM9Y3adl.urY83L2XNk0V9mn5Zyas9bWFx.YGgGCw9dmn+mZyb.dY4TP.T1A0532S.liqD8yv3FqGV9A5YJIqsS64RMxJxgiRCIi0.R1lcsKi+pCOzyLRAtr7EnTw.bjuxTCzGYG9i2WINZw8oD38OPNJ+xF9Opi9nw6+8+A5nUqWdoW5kvW+rW11+5ekuz94P8N4kWw3...v8rl65xlzTm0m+MbzGwjGlSEvrl0rPud8vcu90AaffefX7j.nCTyxfrNNVZRcJJFMEcnj9LBNwvVYNWMmB3TyK3UOaspgQpXNzqoESPPtzPxtVOSsOxFNcFb3mwxIVwVxKeL4F2.15qaxOTj+rg9NveK4tAf86lHv4vBO8qqtLs4LCt1lCsIcvRTi7REvoQNtmITi5T+W+bV8QIQDfs+AJvSJxz2pZpwY+xHj.cuO3bXnPGj.R1.gbp1MC0+7CeB1dFv3Qq+NmM4A+BaAD+axxr1JxyjcpuDrVR6CLCghrNAa4.zWMuEG5Plu5ZfQe9lxDx09LrmkI4gylrN8yFSP9BX804rq6mgwK+jwF9kGHSvn3.gJPyecAHKzuheqEBju1OK2xdHOny5uMVW2vccPm5l8ijMYp+igOWbbHCnWWiVyNXGQzwzw5Q8G7OMiYMK7Y9bmDFlkZF.3Ju5qA+vUc6um6cMqY8CEfuLWdEyR.Hka9w17geQWxkNzv8ae7GONzC8vAPXPC6ooNpHra2QECu.PiRQpWe+TQ2t0l+kVCZytGyo0SmjjbsLnQvVQwew61rmdzyraGSZPKkLrRexfKktQm7yDe9uCV8B4keE7SfGT5YBfbTalBT1SSQDnv2BwP68SV+dLEm8pfeOkXG0vLrnohD.ubA92sDY2FeK1O0q0ycpTocFeVqW6qn3czvaEibyg5gJ2Q+1yU7mkuWj+A4bM4ui9C8W9uatmvv2ZLXz3eavG37mREJeKLAfbdrKP4ujkI9HN+gpFEDiylmPmzbgVyeCk9gIzMiGI8G8R.x9lnBwKypiQYqsG76peSqJoOovG8gmN6U9eZPPMMfF8vJNMBn.ZswO.odI7I9DeRLsce26rkqUt+ezlv285uwa7xOuya4CEf6DTdEUF..ZdiANiYsOiaZSaZu88ceG4qnTojRIbnG1gga9GdSMuSxS777jVGcXrDIMoPXP2TfrGobzgbzOFd.XBPCdk7rlWBhn5pNWN.E+LsJpNx58UfG+z4m1yrprI6zHlgrbGxTbcynIQwx3uhB0jiwZmo.VVyQiHf31.iDqzZ4HpUXwRa1xvu7uIYygY6qBjRtzPaYTHo0yieuiM.nYGqq8UE3KCVjOya5QRZ3dODzDQFrHsQo97Xxxyxl.bGH7FCjCYzHAKaBhsHs5Ii+c3Wf26JpBu715SVNF9s+lDwaSejsMB8quN.IVa6rXznufeVV.Bd2uXavTe2eE3S1dxgW5PAdldz0plEfo.cQevlKX+FuudZFVmJFZsH7c8ojGOM6UlxiaY00Faq3hz+oTI4TPpPexZ86FWniIH362uztlSNIVAPfdBw8fLQ3sRDf3rOx389G99vw7lO1JsY2kW3EdAbFeik8hK8q8kOvgBvcRJuhKC...WxE9M+Oshq3Jt+G5ge3gBtoMsog+rO4Ifd850xa4lBOvm1Da.cYucfEYSTwQ+H5Hq4EdbfqZLfdrB+.7+UweeYS.ZsfLuxhtpCmZD7qSXIBHh+hCIQJJiraFm0N417eoyvy9smLm39lVE+ybJNqWk5PSJqxzlFrQwaSOYsM.UTBHahvleQ37xuUCdVVk46ffBrpgy5ie5G9Nuo6j2aARD20De+7BeJ7L9NTPBbSCfqB+GejI+X3S5u0pvOKKa4M+XJDo+3eouvy+Y+C5z4gpi+ZCOS+9YeA3KiyStFHULFGGXWgGbpuJlcUmeJxUeK32jcHqG0ttJLWnNDIJVUdzfOE5+3yLQNP+8K+eOB+Bs2cVXLBSjeFjoPek0FG4Qbj323272bvsYkxkrhuCdvezO4WcnAbmjxq3x.fTVycdG+2m9r16+Odiu9iJMLmSy8XO1SL9wOdr90uNvtpWaPZ2euAhXcz0KEMCR6UtM5j0LSu0xnHX48dfMTMosI.sFdlU3.7sUFjyky0bea8U4oQ8Bv6y9fUuH9cNBPn0k8fREUJyQiYi9I3bqSYhcVPvkmGi6OfQDdF+AZm74yURgpKJMRHC4U.apjg.YeBvYHHk727fIvzkmlhWJMYX5ck6f.QmkHSk0wE.tHpi8e7Z3qrZJY2jj+bBuLNRNO9hQFKyV.huXtwHD+GOtdB7RD+86W17kZD9RlYDYdO0oT2LBgepYuvMHgk7Unv3GRhCJ9Q1TEgjx6lncCsuN9EJOyYDPmupzuQWNRgH1ZY5P9f9aQ5ufW81QjzSH3nq07mgWwuSXviQy52MRLG3C+5wmfo+RG+L.C+dmBZGLQT3HNQMi8XV3y+4OIra61neekA.bGqbU3Rurq5KuhK9b9eLT.tST4UrN..fsOqEL26.aK+gVxAs3gBvEsnEge7O9Gie5C+vP1w8ZZqRQmAXCUhwrVZMn5W9Mmw2jsKpMKlZ6GMrwFjjeOMBv61jhkb4lKVQL5OQu+B7vyWXPBAvsYK7yFGfoDyROITkAlAOqgRA9SwECNaLR5aRTaNrvK+06KRcmA.8cpcrehh8I0C8KNCHF+AfkVZPiKTog82LjMKXyyr2fp1INPdF+9KPEwjy.73Sw3oS9Gc+PjeiA3E7aot2GrXSJdSjzx37jSNDweP9QcjMWVOQi.0186nd+arnrqLX03+pQ4WwahDLG+LAVXoGjlkClHSOOKNFSivBFSYBI4Z.A+D6jLGT7jO09IpoE7SS.jrb0oM2Pesv97y7ze186b+h9JFNAMPI4nUyYAsqk1KteqZFSQjJy2vjhiabiCelO6mC6y9LmNXx5kG+I1BNqkdtOx4+M+ZuigBvcxJuR1A.buqYMaX2l7Ldi629svCd1ydOG0vkRIb3G9QfUdG2Ad1m8YKOCPGVjrnZZlO2L3MGZiA09li9h5Ny.ay7BKJXjo1Kwigcqjm8iL75iCJVJkdk1nub4Y3L1amNfd0vgHSzIwY+uoDPwHjP1TaoFpH5LS+lw+NAHQJECDUhbjDyA86skksgtMTtJyZNYc9AwTS0Yyb7dEnmZvNCwfHzLGHMlDAlo5UODVNZjWoXV1BXGGNwnt+TKXemcrn1woaXgmk.dYsm9SDeJNVH7OeRKLYUeR9k.x80HLytVE13mXeDK9h+0ULKs5mB8wHS+F2T4w.7ruFUfuk3KEz9DZ.1gPtA3w3YBOpw5lFVGOAXueFx7.ifC.F7nzevYfqfCQWgRMYscr84iw+VVRJ6IghweKXlt7BwnGV.DqtseC77yG7C8gwq+0+F5rsqU1912NNqkdt4q7RuvErksrksNT.uSV4Uj6A.tbIWvR+2bAW3E9LO0S8TCEbSdJSAmvm5SC63DJpv4MwUtbYizuTCyvxHMfTJ86WVGMVQdRRkY4orRorL3MoelWuO0tKn0G2QJd5pA+YWnYhwdcc3TPJmy1BgkE4gcPpM72nkNX4FpBAP3vOAkbnxQphgdn3ugT5qQxnMAo3qF7pxmLgeG1oUqOZPHS+shwBxtUTuX.+8f7dHf2GA8R.H0itTZ7E4hrAvVRg3XsLI+hsPh6WQa34aGuZGk6gA9H9yPtDdpS+79.nK9G4xkyTE4GaCWRWdf3qa7WHNm2eYsNwoOY23Ou7P+awfteLhMmsM7TtyRcAODueHqqY2scWxZ.BQL3hKT1Nt2uN3172bN258YAnKWL86ghlR9NTApNDW3k3otJwxIH68jr9Y29NPtreFP.WNb6DKx7+15u6iLdq+p+p3s+1eGip1kKW00bsXca7t+ytu669dxgF3cxJuhNC.RYhSXbesm4421odzG8QlhufWFTYZSaZXu268A29scqMOHwSSaFnyoDSdlTWQYRsiHXJDdl3Ussl+suC0sI2A3cJtnIw12fAZG7eQuPJ4uK4S.5t21x1QxE8RabE1y.FSGRUdh9epsHGrT9WkkNgn8vLAOQ+0fmM1GnPB+rcBm.1.IZzHZbI9YVl0p2giLQbPvud1RT04BcJzHGcdOh9snzk9RIOS.o.E3yOSbM2Gd3AZeJHZ12CAYcf9arIz2y+fRqu9J2MNhiD0sLlOfha.gGP2dFP5+34Pgo5F3V6vsnvyd7lFcv6vumv4Lkw7guMEnZO5Wpf8dc.kzra3uICgoAbF+IGtxz2awSIemE3eu.GwG4.7xoCHfpNJYWaADza5I.st4bFGvAb.3S9I+yw3F2vYBb8288fy+BunKe4m+xNsgBvcRKupvAfG3AdfmclyYN+jdH8Gr3EMbmFi4Nu4gssssgMdu2K3AQpW+IScYzPnYrqCsPIZ8wyn7RknXHH0DINahPAiz7npqSgo1rwxfyB0TA..2wkxYflfQW+MA5Voim3s.+aSVqP+BRbFJSFNzVvpmTGm7uS3IsNB7IO7QKIl5Ai98lbbUppA+Jh4VfqAvnuhhKpuEicJ8KJDKNDnetoNwidn10TvO+aVRerKiHNxe8cEAYvaXfOh+FC6V7md5uuR+sNUEjfLS+aT9E69Z0mvOK4aflL.kT7w9ywEdYl3SOP6T1aHx0FYlWrTvWGdn00ZCB+kr.zv+7XQl9SbyXYtHPK55qC6EJUBPONon74lnsQkh3nTx8LcNEiR8WYmqL2Jam0OZbV+9pd2Qxve6dPeay6I.UVVp9zm9zwIeJmFl5Tm5.wRr7ydxmDmwY8Mepy8r9pG9PA3NwkWU3...v8r165Nlvjm1wseKbgKdX1O...GzAevXSa5GgG8QdT8YwaIvrnnNmQJEtdTTajQUJvY3VdQZjR8bOSiFOIomNqQ0lB+tEwFgCxhmoRMZkxlfp2Lfz9BvvuDMGYdjLnxQLx+t8bqs.qDijiwnL3e2ISC7OFCv6veP1pOkvEbsEK+nJ.5wb03elcBPEE1tbmUyQ6JBBWlyhM.TTiKm3.8dP21c785kbFua1471m0+RmXgbg+kmOpgOf+F5pWQrFo+dJ8y6x8XD3IyFm22f1CkMYesmodEIRUJmGr7O66eiNB5Mbai+ZsDVLCnRjng+H7vCuxzgwnIOunzuN9EDLbSYyQEZQtjmZ1HvMKsPyyoNcmkWqCvvaj98zkSVpCe8mXEfxkNDgK4MynlMRm.hoHM2Ut4vryOw6g.VV1qWO7Y9bedLu4O5uCY.ZV2+uwYeN40txade2zl1zyMT.uSb4UMN...b2qc0Kc2l5zN4i7vO7IMoIMoQMb8RIbDG4Qg6bkqDOyy7L.PFL6MnXZy8+tkB+1N.H.KSLsSGfAgrY7Rt1iTNkn1lLFa3WQDTiubkCzUQucyDNiJB7ZC792bbdCyN74rMlrFhCdQ+M+CSDqQLLpTUmy.0fWMdyZwcNFQNHkBxKmrs4AbjbNKSLN5RTa5fQEBvAJbfQJ7QSmkw28f7VqymAATbDn7JZUk+syjfTGygV4XVMDv2B+YyfPwfum9g92bW7eEYTUC+QYqqvVHCo4W+P13+.BbNHnDq0BZJp4oLD84ZMEo4QO7C.+hQdaMw44uDAPBIMs2IT1qO1XZoeWcntk7jLzJC+hFai3Gd4NjOQ8W5wmjFCn24GTcbdLnTT8SAhgcY7b1ZGp7Q+XebbTG0Q2pcGoxUbUWMtoackezq56r7e3PC7NwkWwuI.ik6+tWyhW14cg4ssssMTvM4IOYbhe1OCl5tu6PTl0rQ6nIfzZSAvCx52gxnRgGDRanNAHWZQAzToVn.UIBuSZkmzzj9T2oSlf+4whbLe52uOcgAYvyu42TL1A9Uu1U5u4YLMqxTVtjfSFqsa13aFdzE7YBdE+gsIlW7o3UdCwI9zXoxNJuAZ0f0ZXgTpY7x0OQ+sk7qK7a8T14fuOIUJFnK+umrx9whE32de9dxejguN9Y9pemzuudsDIVoKGpbxSaVB2JcI+hD.ajHqHPFOU30P+UFgG.zDIc.doOM6nqQA7IF+b2U76PMjKi+EQibA5D68TAfNu2+21Bp15ab7Cieh.T5WXwrMyVZvTBkK9preG+qsG7La.djLYobScp0pt2L3W+232Duk25uRqmORkUul0hq56d8WzqjdK+MZKupJC...aZSa54l1dOu6bau3K9G85N3kLTvNkoLUbfKZQ3l+g2jlFo1QZ62e.MkDEfXsvUfq9MJx4kRnB7oFuc8q6dwC3hlQqpVc3q0WMHGtA3BsnxxxBno2K0r9foDbuHgZ94H9k+MCIxb4YwMqm.NmAAO+GaSJzAFdTCdqApheglKNN3swDI.Y+ETjUDMqvDWmAF9AH1cHNW4YfZRO6qzBS+V0xgQL7hJPqIsiKxg5Lxv2I9Crej9YdqE+W62ckRimIZIwzuec6UeEJDlN9kPshFyyqP+WRmO444XVgL92MnIQ3zlZnx4tg2ZC2xUTZfVzOwmfZ4DPqM6GfcBjpcD+Xd2I8cyYr8TPlH.V9yx..+ELDiC14CdSRGK91VmMWveR4KcdnCaYbDG4QgOxG6iigYShC.7vOxify5rV1Ce9eyy3XFJ.eER4UcN...rw0cWqeRSYVyYVyZluoEL+4MTvtG6wdhYNq8.q5NWo9r3ZKySrkeu8FELLPiR2JfLeMaNZnFUKqWVn8iJM4kIvsjAA6X.o1FakhL2qrrDhQu9fuW.j6RfjeRrpYBU34b32KznpfkNEAprrhgb92GqvK3Ox+llUssDYlwKjiGISIXf8fsFyMeSWC2n.h6FjOyVrPk5T9N6qQ01Iv9wtB2yyT6AeaOpfe.3WKQC6w5G4emLv5mUBLH.zeMWQ1jayfliyQ9O1.bFlL5JED.5WU4m8aZjwr7iImx+5SuNCuu8ajSTC3981dRwANSgCn2EHZU6v3uw+zHbdLbQ+iEocLs7YBbuwZ4M6WC8XY9LyByJE1YiVxGFdG8233wBl+BvI9Y+7X2lvDpyvcTdgW3EvYblm01+J+yewYCfsOT.+JjxqJc...XCqa0W53m3T9iV7AsnYOyYLigB18ceWH52e63dum6QeVqMUB.DiG.rZg10OV3eiumqUE.rCFt+XeI4+P4adGQhMPsMFXgMZnkDeJAZaYpWrMI9mUJjDEVvYGG0rT3NNfjgzznFdiEbFh8oRPkOd3yVDdBtbNUvkTjXhMuyHSq8gQzINx.qqgpMrggKpuuiOmpASktemuIAmBFH7cg+ZkA4rPqgiQCAs2GFtlqE8G5+ijmCelmb53GxQHCkdAfh+jG+H42yA0kOYkmBjITCow4+D9a6bPCB5GdNu95ZBxSopmqeSJwGGS5WH5W2GBDMyerE+ijtg+jwTQi+ZMCN4X3OSyYjMAXj1K3O6kqyXlyBmzobZX5SaZcv20K4bebtm+EfaZ8q9X9wqe8adn.9UPkW05...vZV8JO8INkY7+5QcDG93m3DGt2uyG7g75vi8XOF17O4m3LFoZGxrR9fQS524c8qT3qdXY4xbWwvD78EirjgIWvvvh3zZAx3o117Di5zEqTQNsB862GbFFDEKw1UkKpxqLbDpSudE7afyB.fbLZ+ZvqBiVvq6HXM8fL8qPinCWcP9pwEqc83mvNbNjvEF9.86LX3fY.eu1uUwOuQUaMZfejv+npX7Luy5MCcYm7iivuwNf0AY8esWN.KoQjmKh72nj17RlqGuo15B+9lHOVgOM.3o4.8IYFyZx7SwUglSuQICecY6u7CreNJdyVc3g55F4KH.k4vt62AcdaEGU.8kZTVXdl1WysOK6UB.XRSZR3jN4SAyYHuleA.9tW62CW22+6+e3pufy+BFZfeET4UcaBvXYS26ZWxSHrHy...f.PRDEDU2bYm+Puo.A.9nezOFdcutCkdRYXbGCXytQ01jnNqGUza.K.ci20OLyxzMkAZoPQlGYVP7nooAZfr6Ib7MFn+MAVpi2DbD+kIi+vTBnnWM3lqCdf9iZ8ME10g2oIDlRJohd4W0Fv+mfXpJ7A7y1uaKmqL9wYExTxWsLftsNq2nElVzxX.9X6Del97N5+CeJ1OOR8eY86cO9ulQCqApM+sMl7yyI7mC9usiFdc9WdDTb2zBsdS90wa0uVRCgNBYCnH9H4GuqJrmKyiE8WFslJzUP2UWbg5ztGeb+m+cmPYxSFn235gO4I7ovBVv9N.LTurl0tNboW1Ubkq37W1e6PC7qvJupNC...abia7ol971maeqOyV+PG5q6fGJX60qGNpi9nwpV8pvSSW0vxlCzRWZ6n2RkIawTbWqHQ3Ntd14yWhnuWIJ.FdME6J9k.JRt1rEdrezYntkVXRifrr.4b1tk5PC+KuO4sr0ZdpyGMxrCOYG94n2Zmo.Cl7.geP32GkRF05+L4mDokuOMHmbNiDwusgGMYZ1.JWGdVN3fun.rK4j0AvcDdxs02I5nUolEKWYjwelj+d5mQqu+ykokbfWx7gJaDfW5rC8edWCKiK45xQjS8+V2eSKD2.eFXD7fmqWa8wGF3Y7SWPWvRstb99E5ieC5IYyCVS3nkbP.pmn.mJgLM+IaTe19NyG5lIlFin2aDDMLn.orMfMz9Du2QIOcoYJrAOe7O9+d75eCC2c7O.vC9P+TbVK8bdvyeYm4qenA9UfkW06..PyKMnIO0YLgINkI+1Nf8agCErSXBS.G0QdT31u8aCuvK7BNk25TwpFaYiwca7WfWUzPNUvugrTyoz7J1g.c4IHb5bJPogPcPGzV3MJnXzsUZ2c7VvADx.qpNkx6tQZ0f2a2bzAu0F7lNTDMsne1gJ5uB7laGjbKv9dpN.e1RIs9mA.umK7qmbRHPdYGPjvi7VvaCe2eaBW+9ffezf+Nn+JTnCJGs4SSuN9u7oQE7D9izgyXXs4u5DRg+MJH53oetmUONk5gF1fmmavUSPOnJx7efzj1v7ooqy1uv9T1IC5Njm4tYA0GSmJB5Yd.QI.AZYVD8HjA6ZxcsURImiOd5fOdlAmqAv+l+f2Cdmuq2cmscWkm9oeF70Nqydqmw+7++6E5Tx8pqxqIb...XCq+t9tiah69ux7m2bVzduW60PA6jl7jwgcXGNt0a8lwK8huD.HE7RjBRTPQifvLlEMRyE9HBl.ZcC8kEO8apcv.OsY1BF8QFMYiHj1Y+tx2TrozGUcYMDkaxPYSJlQFxcZX6LhXQFxqguJAXK7xWUxifOODvGrWAR94xuJQ+siplkIF8zdycw32K+EZvPGocJBejGIjjHxo8xQPqycWzHyaUvusDUwnOaCeb4L5D+w9oVzOqvN6vQD+7HxVQ.NJg2X+PCjhzeaYbU9e.3u0R9H0gL5KsgF0eWvqLRyuo2zCBYTfmGG2uucejz0pbvNeXqDAcAEU9UMKNDcDM7Ka3Woch2texIOfcFwsa+aQjsizW0wkoeuHayg4z.Y7Ndmua7G9deecw7cVdoW5kvYdVmSdCq91OvMtwM9yF5F3UnkWy3...v5WycdViebS8SrjC4fl9zl1tOTvt6SaZXIGxgfa8luYr8samHD6H50LQU2A957ZSan2rR6hdN4a9BctXs0eu8FJr4o9c+u3LQlZSyPIxb6zdSEFK1jtD5k5QJ741IWdYhHsQh1nRVy1JZeU9IxH62aqDpM+6Ll330fA9R8bQcS7r5LG2G3bPZP3mDfjCH4pzcc3UiwD7t7h6EiJ8Gskxsdai4swuvc7Y61F+DzuNJv+HQ+lyHLcDwuAdj.RLhUJggm3e25J3ZfVxfLK.KUIduBTG7B9KFiXZv4fkJoCcjcAeRdyJRzHrkwmwSBz5p2KQ7brDk6hCiIugY+OSyiBfSeWVhvTB16Sj9xEEEU8Nh5mO8LN4WhgO4gOYxpb+LNti6sf+3OxGoSbLnxE7suDba2wJ+cuhK8RtsgF3WAWdMkC...qZU21WXBSdZ+EG9gcnieRSZhiL.TYlybl3.NvC.2xMeyduOQzv.o1MNXr.SmCRkI.57wdEvFvcCPDOrQGtZYokxDLrG4HXDIPihG7Y3xPQTBHm0X0Y.l.T9maeVQMS+PMpFguoY832Q1t5z7fb3GqDze05P+PU7aNB0lVbcy4AAec7ajeg9oLOjbxkTGzH0tsv+..ONVvUwAg+JHgn+n4vZ3mk2oH+iTvY0ryOCGRH7658YdqMPttJ4eauO.Zi+HCTGdLlgW2Y+zuI6YHM854X6HsVVYJywjjy3rVOiTKXo9Y9WnY9DMADNi+5RIV0ijf7vvq+zC.UNI+VJIzUSlGNpi5nvexe1m.85M7lztlu60iq4Zug+yW12dYe0gF3WgWdMmC..Hu6SYhe4G4wepS6HOhCKM9wO9gB3YO68Bye9yG25scKlgb4urMnfAdd4.psbAwPKzkDnroVTGARlN3VNDz.AbF2jmn3m1.fxjnH8l31NPmjhF+aNvl5zGMKKPlzGYN73yNRc7aNmnpLxd5ulQLasgoyANqDiyNPJz.L9czUldj0+oVQntbUgpq+ffOMZfuN9KTswWUnewQo17OaIMhed7iHqo5mEIJQeiH9M9o0Qei3CN6HcheQ9k0lG5X6JnWnVg+YiYtL+jZ0npfos+LraQBmQ8yUvuw+.9Jl7i4M1WguOIu6qvmKu.eRpi25k5iTmAD0uqqNL1pUbHAGc7sYpJLt6b.Yy9w5IFg06mme61DtY96rrx63xRNnkfO0I9YwvpKG.3VtsaGWvkrhyYEWvR+bCMvuJn7ZQG.vl1zldtIrOydEO8i+jmvQd3GV40z6nuLm4LWL6YuWXU24cvKiKrTvKCXiF0smMRmN.mA4xM0m3oc+xLBuwUDLba52nmnVL8Q+K+IZb0CM.7JDxM7sbRAZzgZa5N48AWuT35D1og09nZ3RUT5+ciXcZnUEBIG+6rtpzqgqJxew1F7xMRjz5agFMfR9eCzeWvO.7W7YSo414zHNVpUKDpcL6EDAPLScn6B+Cl9cFeyhA6Z3mjeIh+SjAEeiBw3bKtOErimi7evNsi8a3jXFDRsYpxuWYwdTb1TxtGQJPf4bQbdaBY8F5Tey4wms+pF+qgeY4tBOyoKn83V1Yt9kLHnq4O0VoTOMKgiDYwNmn98w5fDIR4286UgF3W39s+3y94NILwINbYyE.3t2v8hy9bN2a8hO2y92ZnA9UIkWS5...vOd8q+gl9dsmq44elm+CbnutCYngeAKXewLm4rvpV0cZyV0Q6jBAWTVASAIuBqVkhwTY0FjaMPQQurlasLwjH0QIRgUvnKqfkmoFyrf+RqgJR58JQE1bcBGLwQDfrjAZFCpXfV9pIJM3aYzV3gfMCVlIJ2bG8OR4NqpKyBGJM2FLdiFryJFC3E.NCFCD9badNz.riPQCrL58jRkTVit3+1xe1dYLx213OPyQAPbNhyHavXgCbh+Cvyc9sfuMC.1lUKCcTZIxTCzJpdm.fvuS96aKarnUhycyYPQ0avKWFWrQWYNWWVX8oo2bovsQC44zYfnSNbaoIKIayC39e80KdAvAa3WI.2bW204q3jBrn8iYUZe22EhO+IcxXJSYpiJrwkG3AeHblm0R+om2Y80FtyF9qxJul0A..f6YsqcsSdFy5kFWZbu6EcfGvPC+BW39goMsog6Z0qF.0M16ljWT.UyQ.2FsITrkCPzk1jwBYS1UyHs7LmS.jwPeHOkIzRZFk5TB4zSejVNN5ibiCJIHu636oQrjSIjHO86kXGAf9C5wgDlx1FC8JCnALAGkXzu7CxRHjj1n.XFY60epJqLb5QhY3AzFRJp2MwMTg.bGCwjoPqK36D+N92jS7ZRCk9X9GpLisA2J3Y3etqGN64eFO0vOy+Q5Oti0UyM4132S+d9uE7X.nmwOE8Xj+q0.sRyOw+B8LH7ydUKqUchjeHmcYEi6WREC7BEzbibVR4urd+C3l8qY9K+DxIvhHwkd+jwyBsXU1D+MhgvIVozf494lrRjs53ljgP8gm9i2Llwz7q86Ehatyad3yeJmF18ce31L2..aYKaAesu9Y+7e8+kuzqYNtecUdMsC...aXs20ML9capG3LlwzOpErf4Ozvu+6+AfoLkof0tl0nOq5ECT3YQi0CLS.vWedSvz7av8WQoLeV1EsKryBoVzRpc6nv6M1hJzq3fB5IlxRk8Df01xFDTWKSmrQnQAksid0mQ.1vYnNx2TaTEmDX5ukbikGY33emSbF9oeQ.2nGmRvQG7Q724Q6SbLfEXNYT.+lWNAqsQv6.dDkI0vOy+soeF8j3eDwOOZSSALCOBvSLVK7mfsDBNuMZS+jMPRVz9DGvKgDu7BfMlpCkBNFPvyWnO5F9iWS8NuPeLpOJiaczBAo+fDf4PcjefMB2WcdNSOyvGa7u6M8GehCDDlTwh+n4BmrNWje6yblCN4wv86O.vy9rOC9Zm4R29UdYWzduksrkmenafWkUdMuC...285V8E0e7S9WYu2qYun4Lm8Yng+.NvCD61tsaX8qactm6LQ4LpXFmSrBgPlAbsUHJ+TxV2cIk5B31sEFf3MMnIUrAQ1vjYRRnYQerMA0GsbcGAfnLHAf9d76UVVVEgRciajQu87f5dxYAuQdmTtBuT2HEK+LgdD+diKwvpUdiI+QK7Q7WST35tnywdE9Ww3HgepcPoep55cqljL4WW32MrfM16sjOpweM9mIt3K.HmKWQ7K+hYQoN8qjuc7LyTaxi6bNYIfyUNH.j4q5sqIwjIHajN3ey8oosnco0KPHenyzTT64IBuwWfN7IbhOJuY2+T1o+ojcSCBSOUGTp8wB9YwGH4RM5uOxXu268AmxobZXFC4K3M.fWXqaEmwYtz7O71V2htqUdSO7P2.uJrrKG.Jk0ul67rPuI9GL+EL+4N6YumCM7KZwKFSbhSDqacqE.rRWSqtWQLaviM1UwfJ0.MSXZfuWO6k8gZfMwoNGNEGC5nC5L1KlOSvaTRLR0xnb8RuTS5+y.t6z.4jBH3rIa.9SSfhKxPhZHJS7Go6MDXK.IWX4moStHy.r5QYpwI9pXHL38QK6qXjfOBTE7ajetB8CMnRuwZxN4n.+xm65bu6geP3ua52aHkMXGwuQYTUb7u0VMx+AQ+N4WD+EYQUGLzmKXrceVhE.A7a7eSk3qvWHi4kgOpA1xROjjWA2jwwtroBa840WfPfmGUpUNtr.DunOmcVw3e2l8KA8x.SeIC4vQccBNYiRg98DQM3EYbNmwbm6bwobpmFlwLmYWBiNKaaaaCm0YeN4a5tW8a5Gb4Wz5FYHdsQYWN.Pk0bWq7eNOtI+u6.2+8alyblCuGlKZQKFScpSE20csZ3M72UDn1uyQgZFhohKcrEXJeMWlHZ2Y.jAwxS7eGFg.AWds5F5SjgCJFPMbg.7TImkGmr2o.p8ARwnKJ5XFLZf2EwUQwDu4rXkxQE498Q.Q+ryCvjUwLyzB+dvc3uQjvUH2M7jiHrBxrqMKDXmDPqFyOPyg+ba7K7+nF9Q.+.0keULP2E4WC+Qb36+C84Q3K8+sveN9El9492B9biqXGCrkkPwenED3aNxbFSKN81m4EIC.CvnuNeTaJh3TzS3mzWnv6PP19WQuBkp9dk1yDUkSljhp5DqpWCj9pH4IvKF6EGgnJLu4MObxm5ogYLig23+162Gmy48svJu0a429pujK5FG5F3Uwkc4.PnbW24s8E5OtI94VxAs3IOr2Vf..GvAbfXZSe55FCzJrwUSUebiqo0dzjI..KJiB7M6XXC9ng7nyEtnBS1l0Jtu.HxO37BAevPnVDkYkSwfbnKEi+Mu8CSV5PS01SDdmSZa7UjCli.B0ylMhzW21QilpRdvc0iLDHJxSiB3U42fv+fn+nvnc5v832aDcGI7s3+.8GodBk9OmiBl16Cjno0H7d52voC+LZT6L79KfvXUGEhGgxl1peM7WL7iDzw5I.cSxBja1DcEmjAf6dyucg1DcpQSxQDgVI7axhbk5QY3fpaJA29R.nPez9QHgAPlUoIA+bsrem28+RYAKXA3jN0SCSeZSe.XpdIm6iu8EsB7Ct4a6icoW74+sG5F3U4kc4.Pkxptia8uGiahmxRVxh2swxtLc+2+8G6wdrGX0qZUHp9tKGADEFovuOnSGPC.Ei+nYxSyQErnnLkTC9tWgmpi.FU4+i47f4v.wGhgB1IkjG1pkrwW862P.8.z6Kb.zb7lHQhfe91OKIJhC00wHTCvQ+nNrnzek6B.p4yt9ARl6Ld5H.miR5lhJ.uuekZqJNDDs6Vg8ctqXFM8+086TJg2Q.OWZK+fiARgJz14wJ6ZexHbicD14CKiCoVvCO7P.iNMAb2m1+6E7s4epi1MGuXnTbTIaFQ44G55mCnoQuR.4tG3bcMSOO5UEfi+qAuqeshEbQtqm4+98McHYXiqGgH+EGp3rpjTFnM7xxDJOeeW3BwIcxmB18ce32ve..WxJ9N35t9a7ztrKZYe4wTC7p7xtb.ndIuxa+V96w3mxocHGxAMgoNkoLzMv9tvEh8Zu2arJ0I.1fOZ+cJJdekHiSwRRVS3vlCDAURofwdQkax+cilXMJT85j98NaXEmJK6oTHalsfhRSMi.k5V3emQZMhAxnaKmUzeAEcJlAKEbwYIx.z.j+ImpK8GcDfrGL3JVCd86NGWXmJHRPnem.L4HUVNpMPK7G4+cbvK+fi+Y52y.DVMYi2Q.F+v0+01afA.OL4WD+IF+jyJASRAJ0idt+y1vb1QbzHSqx50jaWdPoE+7GyeiJxuDW+fQ8Ta38iH83yb11uDG7w7yQ+Ch7URnPqNdM4zW3cpHiCbwKFe9O2IioL0g+b9C.rhK6xw0cc2v+wUbQK6udL0.uFnrKG.5tr86XBo+A7S2xocnGxRF+TlxjG5FX9yeAX+V39ga+1uMzue3EHDUzf+HsOsNJgCxQ.pcKycLmCB9cqsMomrczydiitRlfW+dfQb7Xa7C.MMhxRCHo.UhHp0wlDYcWQyNyvJc8YDvi+nR8Xv6rwCmyFlVyJ8Egnx3HZCxnZv29HxwqqKiei9k5YQ0RQpFYpH9I4zNT3cNuzV9I2Lb9QBE4WUzGwej+88qIzA7A4GieWzutFHP+HjBaaMZ.x1q71lz4ix02all+0PkY4D6TRgttA55z3ufP.eT5lw+1OudD87yba.RJ5a13qeyXVh9ufB6DCnC.BEoOM05YQdqS7my3vOhi.m3I94vjlzjFj.pyx24JtJb0W809+6x+1K6+qwTC7Zjxtb.XPkMu4ss0m6o9md7m7YOsC80cviaxSd3cBXu2m8AK4fOXbG2wcfsssso5NZJh26s8G2m13zn1A.sjgk9wRaH67X.+NtutC.rtNqd9Hr6B9bU3cEWDJMZt8o+OdGBjb+wRMecmUX8SIRFzJrWRtwsfeSq4wejaqwe19onC3IOGXWUbvSNHXiPZK+c4ulD.svuTWmg6c.vWi+CN3Ti8ITUS.P3Ox+cf8H7sjeNeJBOLL9AT5y45UHVwwSywsBMovzf+Lf9twHAaLNu64aWJFyqrd9pSAU.OyeJUGdcOSTCdWeDcGDvhfQJpesdliGsAgcHv6PE.vwbrGK9DehOIlvDlvnBWwxUcMWKtpq5p+6tjK7b9eaL0.uFprKG.Fgxi9nO5KtkG+Q9hO8yssS8PecG73FKdjtG6wdhC6vNbrx631wKt0WD.jABJ08s1.RINxTqjpnvJVA8Wy.M2jX8a1jQEEW8b3mRcaqnOYi9AGThvSFb4I0suyCBzdFstcyZez.yNkSYpoDYD+hTQoel1b12rM7nJqi1UcJ3ydpl8voB7oQA70wuoPLo8GDpHYrOp135TW+2TzSzzO2vm6R9YzumAZXNi+iQeGk+s4+VOrC3Qj1pf+VqGtyw.kA.f8NsnW.8xtjWNAKZ+eIaVxN6W8TnZwmQDF+p7m3+n7OloClOhu8RAjnt8vK6GGISEJXcrV+rLMdpBLmF7zZUZpvqus296DezO1GaL8V8C.3Zt1qGWxUdEm9xO+kcxioF30Xkc4.vnnrksrks9Sm4te5O6O4gNkC9PVx3lxXHS.yXFy.G0Qez3tV8pvy8bOm9byXdtX3NZ7U9hUeuQknyATpG40Tqb67INGnd6WZe6ZE1vWtX7z4D.6b.AuTWO85MP5.LVXcGkzTj6KJ+Li54DuZpMefUF2iZe9RLgMp6+KQ+rgWO4y8.c72fgahlpCumdbmscUQsXMfZm1Vmc1BYtpF9i111Q.erAL4WKqY7.Ap8rM6G26F8YnB3JALZguF9YxmoeYCv4FGCJZeX0WVJ.+Z8mKzwfLdBsNoTj7L3FzzmTxN8Ar9.mCNg4sZchze4nD2rV+ryDcLuszns4Q+N8uc1FcUE49Y7676+6i2+6+e6nNKCwxU+cuNbIW4Ub5W14srOyXpAdMXYWN.LJKO0F23y+DO1C++2O6YdwS9PN3EO9oLF1XfScp6NdSGyaF28cud7TO4OyTtC3FzyFKUCrxrQYJYxOYuVo1RBzKA5xCxJ12ypRhjmXzm0PJ75.y3xzZ3yD.asAAGEBEJJIdeAvzeJA8cL.P1s7AxqhXc4CT7SHHYJyapUxPJozmYK9mXG2xUgWjK0g24n.AeLpd1BkDedVUVKieHGEXbzB+gVOgc.vWAnVLYX+NTPWUzmQa7G4+b1K+G.7Q4mennj0q5MPRa.DLlUhlsL9StXqzWHNECZ.XDNNeM3SiVO6eJO6vud4jroPytT6WXRykkFXzksPWGeO7RJ+iqbPJJ.HGhZk0DHN8D3xv58yHH0KgOxG8igeieieypRnQS4xuhqBWwUeM+Cq37W1IMlajWCV1kC.CQYKaYKa81GW9uG+zG+TO3kbPiepSc3cBXhSbh3XN1iCady+D7nO7iXSBhFZSlAAPdvqQIFLB0owTW607W6hCJZnJYovTRadNBOzT5yVwziwm5+f24CNBDWZGGA5VogdIscjS5.+NF.p7QVhCYCYw32LHYLt7GQgF+8r9LWDLULhKQ98yG7fj2j8.gu0ZR6z9jm+UCGo532ByLR.iQ3AmgFO+yoslC.MlT.lMhm4e.YrBw+D7AxGbOnN+gF+I3ukuZkemeA8zmZNaM7o9xx+po4WpezVY0BQ.77Cxgxb1dBuoDapJaD17ygaWgdh6WgH7584uNmevGwuVdqvbkStVay+400MwINQbB+4eZ7lNlioCbMxkK86bE3Zulq8u5h9VK6uXL2HuFsrKG.F1xl271t8a8G92hwOwScwKdQSXZig6IfwO9wi23a5Mim5odJroezOxabjzJVeyAFTHCV42.LlxgqWLnlAZ4HPuTB9qjWqYSs+Bo7JqF2IlwpKE0UzgkQ08cfpHN4tzTryYO8lGDP2CA7Utpn7lO4.1MQXaRl+2b7a0h9dXfmLP4vO0.rsIGpn+FaQuLLZJkqasSgvX.dlVB7uaXoqALfrwxp4o13m40TIBxpg76F.VEdG9gsl87Z26MnB5xvwO9geQ8n6n+AFweSK29L5K+Vg.CDfJ9TV0Zm3yhNE3n4xb.8r8irsLaE525y5dtXM5u9RXj7PEF2NsoMM74OoSAGzRVRm3ZPkbtOV9kd43Zu1u2+wK9BW1+4wTi7Z7xtb.XrU19cbq27eW+dS5jVzAt+Sb5Se3ujJ5kR3HOxiBia7iC285We4oEkLINMc1lsqdJ2gadVJ9.pn6Z9Bp5Uhl1hDH6T70UVIXEqb8TyIjBedSdESysF0N4LQWNInExQ.405qvWMJyBFbIqSxlfjkSr7xsmA.0VA5RMZkH9kY.UfMH3I7SQeK.5xvRoAhFx3npag91fC+RdD49wB7zW5pA.EAK6HPEJH2I9qv+r7W.gMn5ZKtwpie0.KQ0P+MKqD73mXVxbf4Jrw4r+qbsXCpJdRda6cBtOhc2KwGY+ATAdg9yQBniRWWbPViEnkj3vFIuPFyYNyEmxo9+Bl67l2HhyZks2uOtnKdE359Ae++CK+BV1+swTirqxtb.3mix1uya+l+qdILtO8B22EL08XVyZL0HGzAsDLu4MOrpUsJzO2Wet6XfoNDz72d83crOfYPlzRTy3YQCpZvTl7mPIp5dwpVRYcSaZOyNhSrh83ECjm9xt1FfZejTXLpe.7Qg20DZjR.k2G4IH6wgd5Usp7tUWhjSxxgpvOYufhzFmD.ZBNXYhH+XGVRCBd+Nt1avgy.RkSLAQ.1FyLJ+oL4HjR.+b8xJAL1guU55i7OaDSiZzN4H9n5qAd4CjimrrI6I+lpFgGA36.+N9GY8kXk3lftuSbYep9PSqz1IB0Xd1LLpUUE+EdrkkZa+ODk+59hnH.DYsczeafu4z.0C86mQeB8d9uFmXzThnSuy.libt3SnLbkyYbXG1giO6m6jvzGCuQ+.Zdw9b9m+Efu+sdy+oW54eNm9XpQ1UA.6xAfetKqdk21e2Kl68Ql6b1m8Xu1q8ZL0FycdyCG5gcXX0qd034egmGx5z4VGU.yKZH6XeKBXtLZ1Es7oE.jhP9Z40TzXwJw55cOy8aswuO6AlgEqs8NOHOeDWZ.kPgs7.8LyRhKUVxkKNafDcRHrc1sy4iV4NV.mLLxxlgDd8Qr7xo7jSSOQ2v1w2tc3czooV3OG9azv8vAOutxU4ej7Fo4lL4QiS1DvO2l5lXKAR9GHQ86wiFJY7LYzE6DswAMx3Vo4W9amEuLJGeDLC2gDpU9kHav6IfrSFYx+532bBpg+5KNQFtG+G7rKptjQbY7G.nwqYmCN.n09H3c9tdW3i+m7mhca21sAh0tJuvV2JN6kcd3Vuoa+26xt3y67FSMxtJZYWN.rCnrlUsxu3V293O98bOm0Bl6bmyXpMl4LmINli4Mi68d1.9Y+reFHlkK5K...H.jDQAQUsltatuNLx5UhIOii3Ac9ap9XIxtRjCx5D1CxFuyh9xb9f73241evTAUuLjnB3GvNBngFS1mFf5JUGWwQ.5+EqNhy.pSAoD8RNw13fYp8Xa00MNRUBIvQPO5fOXPkTj53yjCMNeK3lQS4qAlC+Qy2wnRGN36F+tFf4MmwYCw9MdXa72h+Chy3KfGa7qUGmsPQlVZv38GgDcuu9IyCwQnzxmoxmU9TkaF+CDkelCT08m2bHP4eo+inQ90Pb7lGbzc7DgIy0GZ5O31IQxeMKWE9nWud3C9g9v328262u0IPZzVd1m64vY9MVZ9ttiU+VtrK6acsioFYWEWYWN.rCprt0bmekst8w8Fm5Tm7AuvEtuio1XRSZR3XO1iCOxi7v3gdnGxOwqLoiiVJtgAiF6kcLeWkHLQEdx+YOqX3HW9x.ZKQoqOKFT5zguMLCZTTkAR2qfY.NBHkdolFVOBgMTTSZbE7yZ3xpyBhCPojcZBTCgYh9DYQroD9mbpvCe23OH.cMfYhLtmPfyXg1JUvuynHS9CC7DNqxJIOOx8owkNPMPoe1i+H+qNhFFKo618jcBQZV1mF3yIC+8Hgq7II6W4LrqqWg+FMF9I52O1MrbEZGfG+disTGQVZa+7TatTYbsrL.TwsYEEpIXvdfkJ3ukvnRyvYwZJSYJ3S8o+L3M8lF66z+e1S9j3LNyyd6qYk2xgbEWwktpwbCsqhqrKG.1AVt60s5ks8zjlS+sus2zhW7AN5lfEJiabiCuw23aBSX2l.V+5WWyCShgwJtiyNB3rC4cDnozQDzj0CE9bIRJIKAIfDEYkXvuYMF4HUhVBD7yzuDBD+6dqRzO2NCHvjG0MjRrqzHkT2ZuOAr0w0xtQQgIZue.7mj.CFSVa3u8lIjLVqzOqH2yKwHsUiEUBAtE9YK44tve6eu5BpOH36B+vLVvQ1xYbPo9niDpHK3TTf+EbZxeO4yvK0NWx3CO+n4MbmMV..MNITF2zc.xD+yQCyNq33eXY7RGYDOEEj7qRa4VB.hNb3uTz2OArg+dov6XiTmLXq2.iEJfeihpzo8UR1Xzw7m+BvIepmJV39s+cILGwxC9SeXbFe8yZqq8Nu04csW60t4wbCsqRqxtb.XGbYCqa0WZuoNie1Stks7aeHG7Aid85Mx.Uor3EePXQKZw3tV8pwK8h0u9fkmIS53MKXKmOTGE5vI.F9D49f9h5ooQR.pQzlHsLE0ZzSrSIw12YbLXnG1umBjZhpr21DimAvebjbz531z+jKJG6oulVMZzVoUIZxLQyx6yc4nSFuNmqRSVWlEoTK6v0bsg+VkKfHY7gqgq3b.hHEiA383W9bqqyZv8uD0yF+Mz2Jc9.9WEzRDyxd6Pt.dRPNtmlg7DAeO8y7KsGfd8nMDJ8dtevEhmU5W7.ljKrmbE3bNGQ+tS9InnTx9lWeZegjg4niM90jwYzLtcPF8UmVzrCP+h33A47msAkIBDImiLu427whO8I9Yvzl1Xay9A.buabi3L+FK6wNi+Gew8XSaZSO6Xtg1UoZYWN.7Kfx8r1U+Cm3tuG27C9.OzG9085N3zX8kZwdsW6EdCuw2DtmMrA7jO4SpOu5Z6mj+IoJU5pdhhlNyPQQQgtN+TzyYHJcsc5u.hZlJAWjIIulq1K8.0VtrHjZqDULNv2i9F9YgQ2N5vVc3a9LYyd0CftSz623X.YDAIzr9wgnjykeVtZhY9zksfT3sXnwYNayNw.UsZ1wkmKF8h65dNZTnOW3aS9M5fuN9qRewMJPQ725tXfLwaF8aveOh+s012j+wKJGl902AFRzvR+hdM8RcAUsMFc3Iv+J9sBy+50wKEUehedvQ.Eqz7Oa7BBNZlb3WcjwweYDyPPaNzmwlVzcKYgW.jczey58+deeue79d+e.LtwM9Nw6HUtiUtJ7MW54tpu4Y8ubfnidmcU94qrKG.9ET4dW+Zt2w2aON2exC7i+TGxAu3dik2jf..SYJSAG2a4sfsrkm.Ovl8Y+p6caO+.R4JaHkLXOvRJoFm4qkW46RgSsdmzT32zSaf9u1lRzH+jRlbzh9ei2ThDMAi2cEtQBWbK7QDzdGJ3MO41E9BIjRHk4HxL4hjCAKJTwXZhpYvoGxwfVhOW9tifScxZ0Xp2G8sBN4z1ffOOh3mEmb835yYOvmReNiBHSF7AKZL4WVcboM7xt1WLHauO6wnr3cNyy+1y0eSc5y3RERm3xne+nSwSrLgLP8Abj89UjOCzj8B40rMKm5xQeGZCStH12.mL36F21T5iLl9zlN9zm3Ihi4MebCFmiP45tta.WvEbIe6K379FuyetZncUFXYWN.7Kvxl1z5drG+Qenuvi8ydtO29tvELgYNFO2qiabiCG8q+MfYNqYg0s10ThJkC9jBeSmz19YNEVtn5FfBhl7xpeUBjQNpc5FmBY2llpSkNB8FbFgMdKskPa14M1rxvFyrrEmbsozBwiRmyxUPomKhvRCIuGBz63ccIBrnBYSDNGOT84DdSVVTDc9Rjc1y7NF3uw5L4mKSBgLRv32GlIuAB0PRGUvG2.dswuOhdv7D0ew7pDgeu.MkT7a7uIEMYsjwF24yWydTHZ+NKVKGyNlGN13eH52xecukOKvKiJrWZQ1y4wktLBn3uMg23XTyauO2bTww0QxnemsMwS1SZ6bP348QFGxAeH3yeRmBl+7WvnB20JaaaaCe6K4RwUe0W2e8EegK8SLlancUFUkc4.vufKaYKaYq2ws8C+Kewsi+3YMq8XOm6b1mwbasvEte3nN5iF285WOdlm4YfpJrintaGARndTzSCrjL7vFda1U0RDxo.H1lDzsIEKejMRZQ3Vwq.GbloV.+8UfwPL71iM0wiNEiZI3LffL99e2m9Yi2bKaf7abXaE5JtRG5yp3LSLs40vu.jesySViwDf9r3Xi5vOZvu7LZ+mEd6SBWRDzmkHpPbpjvpw+rimjwTWedhBMdzVrwOtkcoUchNdx+tAOmo.k9UerDoTJ.o7Ia9CRwdmrwZcviiTT+9WE4sofF5zbPM.seCBlR33O9iGezO9+dLVyzIPyw76arzyMe62zs89W9Euru3Xtg1UYTW1kC.+RpbWqZkewWb6i+v5kxG1Ad.6+HmVtNJSe5SGG2a4shG+web7.OvloIwVDIwKxmQyREnviQgCAPLvC8zBHqWNavW1zbstDfnHPqE8thiVQ1p+.3MLn3LPbikMhv61eBih9iL8+z8+tw+VzmMQllbQ1KQqJFCX5WOIEbjnkn7RodZFHD9PiVlYQh+ghWO9k+23+ZYena3GM3WdlQ+r7m3Oh+kcouXXOEwO8V1K2R9i1afuNM7aDIed0c1qc0NrV8PAGxFljkm9hsrTYG+GvuHyy9L8.Y9SAN8sMXFZ1NZ.ajWmekaB8QFMkB7OchMRhbv2GmyYLsoMMbB+4eZ7q81d6i4y2O.vO8geXbFm4R212ecqbIW2J91+fwbCsqxPU1kC.+Rrb2qcUmO5Mk+ms2WdzU0wYd969dZAg1QRHDRBsf.ABvFP.h8cv1XavlXk3DmzoSxLo6zYomLmomo6+XNyoOSOyzmYNoSmYNyzclj1wY7RrMaFrwlfMaFvrYvrocIP.RBDBjPaHomz6cm+38p59U0stROI8zBP86br48t26W88U06p5asppiGzZKaN+YNC318Pa3OrvBCKbgEhDRHATdYk4uxeCLKrnhc6doxb1iNwE+urMD8HQEDL3f3EAaxG51ILs3r3JZkmjvlxXQmTEUlCpfy+JUYpPjqEXm.Aj9S.5kB66.Bo4Z4SLFnHBMArNnU.3Wi5srU2ftF1MH+9PV62F7lm+cCV+gOyrgnAXAX.k+tD3KrNzjBbQq8GAQ5YHX3O6M.4U7ge5DycM+2Oe93muCrqxBuMKW994ug8zz.Ee2QXH9uA5GrCHGp3JW3dBjK0hJMdfPe+we.54vAScOy.Cx6FA1kK40w.6cVmCWgcYklZCRDdLMoUtB6Zzegj9aS.j+rmE9Y+k+aQlSaZAEucBUTYU3Mdy2toW+W++J4Fppp6MrZLMFTPa.vnLpphqdpHCOgidy5q+OY54kiwDGFgLaZYkEVPgEhqUSMnsVaiecpRS6Q7kn52Pcc.XSooCfWvdRFBv9OWFFBS3y4NKp.zIGEXEUFIOlI3diw5ah0bfj6MRJafoPKai+TaKB1HgXC7Im8+U4Bmj6EqgUDBrNGFH4zFhG2wreonoVfcdzSBDtMIlt6FZv+s1ZrkcSVaAnHRC.92CHBZ9Km9CHDADep5+rUZBYLhO9QBcwfq.9DU1I7iKfvKVx0zgPdvCbAt9b96bVza8KDg8VceN+UFoDSw5mAAZMZhO7ajOBT2.L9GLd8Gf+jmypNDn7zP3dhE5mH8FtbgW9k2NdsW6aOrB4O.vmehShctiO3Du6a9uT..7NrZLMFzPa.vX.pt5RuwcuSc+plZosezTlxTlPRSZRC41JlXhAKe4q.81We3Z0TC+57bvCKUHf6cn3R9ALExDPqgf9SYnXJHHS6xCQq0lIj+myZRPtbQcwxTrc4SJS3COjrDO1rL.Aj9oJ5oFQ.K9xZJxj67o7TL9DrP1aUlQQfsrzBvHt7SCwK+69SCfk5.CACDrJ5NCgyydFiMI22JBAzHF.twRhJtM4xzfk+LiXXxOyPHC.G6+bcLT8nAsBeNE79K+cbYO3o+PaYNjMdyaMxuOr2eA48Otw1pnmTGKL5EYjnm0z5tfcb8Jqj2OeBlAFVH+oC.Ri0z9R+Qef6O4ImJ9o+k+aPgEtng7eS..3wiGrycuGbnC84+m2yNdy+jgbCowvBZC.FiPKszROW77m6uuqdM1bDQFQlYOLBilKWtvrKn.jWdy.kUdYnmt6F.jIsD7sEj6IeMX661mZPwezSRG.uUoyO4xxKX1Zhmpg0xf.KE5Tk3BbTVrME6KztkR5IS3xTNaEIDCgGVUXekq8.GGSb.1BeMSGU.OL4g30vvJz7DuiMLjq3cxmMjCwu0+2RwL47rGl.RESmXA8YYzkuAC+8YpT9gokxe68+fdHDpbkWnfWs+So3yZZ0mUFU.B3+IgsG1g0vupWEjnWXM9K0mXFWwLhh8L9IWNDB8Gn+QBU.oQ2PJhGP1.EQ5WwJVE9y9K9wH4jRJH3uy392uY7Fu463qzKb4Uu289d+9gUiowvBZC.FiQYW8RutQXQZzRKst1YNi7Fx0E..PxojBVwJVIZ4AsXaOCPzqX6J68CQuU3yhZ5zyqF11neL8aP.aoCxTDQqLb17ZroA4dB0O7m67N06ev3gI+YLEnmLwlgUWjpT1PpsXB.sXvDkGxyo7yAOndAK6QrKWhaYsrvFKt80ZxokM9xxkNeoGx65V+FvLPisYOI+4AE+cP9C9cZOaiJfNVJafivMT7TVa5QhQ9h9ZN64XekpPzZSmx+HkS55UQO.coRZoHVt59oQKgO1YJ0tCfxewzLXJbc13kH4RlHE3lbaUBfXiMV7u5G9mgMs4mAgE1Pei8Ave99+cu861xINzmj5wO4wpdX0XZLrg1.fwAnhRu5QiaBwenJqs1uaNYmkqXhN5gbaEd3giErfEhLxLSTQEkAOd7PThJWraDkjP1.A64ZW4NKnCf9rbtEXhM9lsiOSAEoD8H7sXWHvehbP7dQ1HA40VM89T5gIUQBoeYRGSnL.V8e5DojwW+jKwe9X8f2f.JnEZH6.Nh2zrqq3Y34W2..RduS8J2UfH0v+L6ziKP6Nn3uxNPP2SgZO7EqbdazHM9y2vlDd+g96GIcAbiDHrm3UN0qcVTlDUr5D8Poba.3uvGYsKo.+32mp3e.Fq38K9vFo+yT7S5+xoUvTw6+9fIV7hKB+je5OCYjwP6.NiAu97gCejig2aG64Hu+a9aKnolZxyvpA0Hj.sA.iSP4UV5MOqo2+GcWaceuDhOgXGN6W...SYJogks7UflZ5t3N291PPIKWYE6ZFVFH.SwuKlrPEdj2OJ1Hy7ReBt2MtjWG49m1BFFBSL6D+8OIu8IHoFz33lcjgJCeD6OBSXGXVewT.HJZBFZnj+l13wPBNoH0T8mkUTaS4oAQAupmQ94CF92uvo9OcYnxTrwjQH4cq5wO4QX49pnxdCda5TKnLL+TE6TIgyP6xOsvJYajW9WdmVFF3++HJnC5braHIez9pz0nFOQLxQl1niIF789d+.77O+KfHhHhfTNTi16nC7tu6NwWb7S8S26teme7vpwzHjBsA.imPc002Ut3E9G73KrrdXmctf7ldtCqTBDYDQhEsnEiTSKMTcUUht8zCWwE0CeKOsoJhYfo.WRwkMupEoRFJidfes2buSoEnHcwDxLVgtqywygpfi9TO.UvegqKosFDO3IgVv1JNf3suvZi1fZHho.K3yOKL7I5UKU9U446HJBZE2C1FTV9YJUGf9OQAEmdCmnmZjFiMx725yLPUDJT+.TxcneIXbgg8CvH0412RwOy.bV5T7QdcI3.0qe6FD4z92uRu9g8wuEV3hvO4m9yP1YmcPKQNgqcsZwa7+6s65TmqjYdn+3NO3vtA0HjBsA.iCQEkd48FY3Qdpqey5esbxIKinm3DGVsW5SMcr7UrBzdasg5pqNv8fQkRYad3STYpRoHnO6.3YafHAHbhCZZcOSoBRSl+LkDh4mWVtYTZXWTHJhcxfE9PQ.dYqX+L.WQB0n.hYTVS4qxyRGLJfRuUGBb9aR5a8mGzNO9OLi5f.8AC+cR9Mct+K60qgC8ehBKaJaIQOhwe4fHIVvmNDFe4mmJZ17z1A5CbcV07SOWBnE2Gmjfp39nFbBg+VPc54DC2O2fACowF.DWBIfu+O3GfsrkW.QFYjAkr3rL5CG6Xm.u2N20oeqe2+2bq+FU7fgUCpwHBFgcsPigCJnfBhYgKYkU9huvykVgKbAgj1rzRKAu8a9ln4luO+Z8mg.bEgbOWf3j41TpEjdsRl8VPouhBEykKWA1UAsVu5zzSnRVHo4kjaTxMH50UWaDNY3ib+mLXQUPILTFXBZUQffNVXXofP91B7GViyxQiP94BJ9K84Ah99m+8i7KjKBE8eX8bh+9QnW0.jfxW5yY8afEcV+9obs4ykOY9SeFKOvsI9vZoTx7vmdBSJz6MU0+cFB0QPf9C+2BB8BQVIPe2.Va1QppihUul0fWd6uBlvDlv.JGCDZqs1wt2y9vWcwq7ua+68c+EC6FTiQLni.v3XzTSM44xW77+hGBWozVKOXISO2bvP8nElgTRYxXUqdUnu95E0V60ElDvRQnk2VxdY.5GE7JUbdrfcMBKuYBwl6lWPTApS.w8DdJ89+WeAZK1jbTaBrDY6xDIC.jbuJyIwI5E0yaX8Y13mjQQxd7pZaWlQufKw1DVnv3FoOKnzkH9JrAZjg99W9oocxV+GR7WVmH+ZhzaKhLrGgK9TiinsC8yDk5LwWPn72BxGiyV7z+y3ymOvRYF6zjjVvj1T9GDfaPizXB2PFpxeAihsFA7ORIW7elHszlJ9y+Q+EX0qYcC6J7G.nzxJCuwa+GZ6OVUo4dp8sKcH+GmifaVZMFywZ1TwyO+Ym4oKdaaKxbxIqPRaVe80g28cdaTUUUInv2xKPIsnfNwBUqfzDNjqIVQ1C7qaxFNXB.CVTAXaGrfsSyI8rFPZSrIvyInk2xiI68URWhL4pMu8kUtStL6K162D+oE7hSdLShdghBCbiKrVZZv9XKQSs0yIJniLzS7H2A4WvPCUJ4s90DBoEhpXFhzq5cKU7k1+Xxusn.v+8m79Sf9g0YaAwPGP73GP3H4MPuvlB+fYG7St9HnFoX0+DM9fOtQnyzm5UMQ3gGNdwssMr90uwgUcFwPu81K1+ANHN9IN0N26NdqhG1MnFiJPa.viX3q8pe6uXSaXCKacqc0gj+vE.3Tm5Kvd10NQqs2FbaHlWPYE6Tu.kM.PYNHYJsBRviH.rXDy..WtrV9f.9SMPfGDrWkMLBLQLAxF.v9rxksG8Y4xDoy.qIeEFPH7mOmLHiS1ljVE892w83UdNg+DNX4wqjBX4BtiRifxiQJ5YJc6G42jN3H0+kiN.anxVVaLEompLmpT2R.fPCHFkGSgPpyXlORC3hdKY4GjmUV4uPZcTu76nvV+W4iZJz8oFFX2n.Ke+oFRrfEtPT7W+UQhIlnixxfAMzvsw6uq83s1ppbKezG8AZu9eDB5T.7HFJ6pW9ewsqIbiqeyas0LxLciXiIlgcalYlYhUrxUAO8zCtwMpUPQt3Zlm3iqgcEQpTtJ67HyXgADb5MDpI.ZJBXqkcVEUKDZ8.0Jf0YLO3arMBgnWRtn401BTk2vlxCqTCX2iZUdeQMJXfnG7qHHn1cJzTjWBC+xJRnc+PH8N6Qtk7KX7js9uc9aidPqXepxaK5k4ufAAj12zTNj9VmHhv.j2erN9i8WI+F9e1.qiedjJXiUTCgC176SLhh2Yn2UvBHxcTUW.72ks5ylllXJolJ99+f+03YetsLr2C+A7u19O5QONdq2aGk76+M+eRuxJKWuw97HFBdWyzX7Fh3U+NeuqtoMswYrxkuL31kqAlhf.0WecXGu+6gJJub90r7.S70EYEWh2SdxdnPwormwNf.LxICGXFCv1oA8+YW1NzZXaUs.lA1K6kLRg3ft8vNqf2RJdjysus9O8ipT7ZPEE45QPdbh7caJFjZTEx7HG8l8i7aI4NY3gL+Ek.KCjTY3AmZ4v+S3mkAMVJ7E1gDIsi04hfXmmFEJZehypfvae4N.0HE4HYnJkITE+zwO66aAlHpnlH1xK7BXcqa8vs6ged9A.Z7t2E6d26yrhpp4G9Q64O7aCIMpFi5Pa.vi33kdou4OYFyZl+O+ZaeqFIm7vaO5lhKcwuB6Zm6.M1zcECgND8fkqdTR4GExUoNjH2QEr8CFvckPWA15a84KPpBjTkHM4IS9UYXfsIfCnTafRcfPjEjxANTRujBWmTPx3OSI.rjGpnXKr7PR9CozqNRA1keS9MkqyDm4uBCDHzGbg925GP1uwpVdbB+9Cve+geZ7Q9EhpnOXxque5EW1ghiQ1Mdj8NJauFPL7+RiCB72e5jVwJWE15K8RH1Xhc.ksfAd84CewWbJ7oG7vW6cptj4hSe5tBIMrFiIPa.viAnfBJHl4u3kU5l2zFybYKsnPVz.5qu9vgO7gvA938it5x+emKmOVqvo6elJgPuJkJAm71evrECSHRHp.1nmtbBCXL.f0xHjKiLc8DEyLHTngTaBLoxr+qJv+.cUSD3ZlvlhR0zGfXJ8N4AtjATzHH.3fW1P7YCkzyV8E15+BxujWuR8e.1uAD9KCphON8VJFo0fA06dWR7kUPeLHe7EC.tReYXMNPY5.q72hdIk4r1xvx3YaFyv4rn29BFSvDeSfYM6YiW4U95H8LxHnkqABM0z8vd16GYVRIk+WoWdeOd.sA.OFgW3kK9mOyYl+u3k21KZLb2JgonyN6.exG+w3nG4vvqWufq5WHL4p7tVwqWRdAKbcvtmyFK3HH7RfJo0fM+.sgaT.rwKlQAhJDr7NjoPQXde6Nt6+RTMRAtqvFojoDQDRL32vRgfvXKmFA+REk.gmQj+gZ5kpTAkxOEp0aJwe9UE6+hzKK+P33JFvTfOTC6XmHhrTFY6vJhuL9XFnDbd5aqWI4ctPeiDIEqwH6Oi7yKn3O.MYlYl3k29qfYWvbFzxnSvqWu3XG+Kvg+zCWykuzYleokVZGgrFWiwTnM.3wODw1+le2Kr40u14r10rpPxZ6kglu+8w912dwYOyogWSe9mDkqP2ZhdaEYGrtOchNaoRPPIn3FRSPswBAv0NP2eAXEMG.DVIA9qY.CaJDbwnWR9EW5gRJcDbc0txMwsdYGnmnLSUX+kYC26OR6JnnhILJLpx9FGzvfdm7DWg72+Fc.mUzRiDgD8zeWTs47nZIi5Oz9AnmrBSXEyGiCxcnAyR3SvvCI4W03mIU9ETta2XXQO9MQRIkB15K8RXwKdwvvHzDAP.faUWcXue3G665UVyOXu688diPVCqw3BnM.3wT77EW7qjaZ47ta8kdd24jcVgz1tt5tE12d+.bkKeIQO8khDf8IsIJBI4L2oHFH5ILsQCBXHs7q.rl.WtUHKyPYCBjg0ZAOvYSfidkKIx1TDnJm2R8Og1RklDomyfqOv1mU8rgL54xmg5mMv84jop+2e7Tn+aQOa7mFNemhp.0nOejkqm82ujLMIn812YiiDkempY.Q5sOFKthIXWKlXhCO2V1BV8ZVaH0XeOd7fO6PGAG63m9r65c+cKG.dCYMtFiaf1.fGywKU7q8oKaYKciO6lVGhN5g+RFjhZq853C22dQokTRfqHpDPNL+NtL6TDs.9jnjbna4spX6FrftznL.F3ymdoBIzRQi+ZIv95CG7MiHSSvW0AbCJjTNZnfddWlOTpXoywjeoBOTvSdUeVVopz2GdzapTIN0SV4Tdvet.vVn4gUw5IGxa53rH8A9LIMOrb4yU9yfzF1yfIO9xvdJNbvf.aOmZ5s0mEFuLQzwDK17leFr10sNDQDCu8seYbkRJEe7m7Y8zXc08B6aeu+mERabMFWAsA.OAfUVbwEjW7S9jaZiqKgEsnEFxJRPFpolpw912GfJKuB90roDi60un2qBJ1EtthBCLvLfzMKGQCCBtWmk89REunJKrVlgtrpJbtBJ+72GLgaC+gQVLp.AjOoJFGPtvyD8EUkWwBCeBJREaWwMlG07m5k4vhdh3KmdGQ42jONImtE.UiKVEqG0PKulpo2Z0dXEdeeJLtSvw5gP38U4otsWqYEonbj.jBugszkXIXhzvtD.hJpnvl17yf0s9MDR129o392uYr+O4Ohqb4R9M6Ymu0OLj13ZLtDZC.dBBO2K8s9aJH+b+ur0WXKFom9TC4sekUUINvGueTVok5+BDODoQCP1iRphWwJmm58nhWUMX+iD8A6q0RS7CxM.pB..TgfIQTPT0uqFA51QLYmHzoJMWFrnGP2NYoJD46YAxQGQLlEjqo1XAphXAOrohlPLtG9zKGI.6xpXekZXAseSGi.rh1hb+2J+8VzqRYurDY0mj07N3fpUuAOBP72+HJwGLzye+ita.ZhXhIVr9MrQr10s9Pxl3CE80We3Xe9IvQNxwp8DMUeg0cvC1bHkAZLtEZC.dxCte4u9qcxUrrkWzF1vZPnXmDTF0diZwA938iKcwKZMgFYFXZ3PoSVJt6Bx9+j6QBEpJEh8G8ACrW3YVFH3HjLH.vzJGyrGQg1RVQFxNkC8K+DEADkfxJKoqicqhSjIuRdFCUEtn3+N7n2R4MeoVZpR9E00xKFNC+J8naXSx6s9bk8jkkGMW97B2Sph8oCnCmp2WbolJ6stjBckgx2hdECeBF4x2I.CLlGe7wiM8LOKV0pVUHOT+.9C2+A+ziz6Mucseq8uicryPNCzXbMzF.7DJVYwEWP1wl7mug0txjV1RKJjV.QLb6FZ.G3.eL9xycV30q+sPU4TCXCDuPU85oizKoqW1HgAUjABvHqHxJFEB5d7tLDWlgjyk.YEYBKCQKojGkDCvomUfax481vPb4tIZP.wiYXJXHgJOvGZzKZvBM0GxxO26etBe6FvQSuBOW9AF67O3Z3Xg64WVn6.f1+8anBpRcCCw87.m1XejkKFLL.YaqF1oIPXWRIkIiM+LOKV5xV9Hyead66f8efO0rjRJ+W+g69s+QgbFnwiDPa.vS3Xaa6a7mlQ1S627LadigUvryeDgG2u4lwQO7gvmehOGd5paa229Rjy98AbVIdPSOI+0CJiAX7PYk5YcOK4TJB4R6EA.vdpDng0VX2KD.ATbxhT.McC9.DN.mj2faXWSUX1UU47CN5o6bdV8Zwzg.qqSxMOc20SdSaR3refQszgrivuGzNvPMr9jnQopNK7aCfXsQXK8Hj1hK7AtuArLbPE89BXTTtSe5XiaZy3od54GxqUG.f16nCbnCeL7Em4Le4tudkqVuS98jMzF.nA..1VwemWufYMy+zmcyaXDo9...5t6twIN9miib3Cgla1eZFEm6Vth2If6wun+eAM8jKYs8wRetfOUABMogzRLTx6yApEk2WB7Yq9BD8xWV4ub34kO+C5+9FcmQbvRunxe9ljDI5Fx4mWdI3IbMEbRNgC.VQeY3FZeYNvp8BkqXAX+7MP1KeVa4D87H6nnXCMLLvBV3BwF2zlQN4j6Pruz+n2d6Em7TmAG8yO48uyMqZce1m8YWYDgQZ7HEzF.nAEQ7xE+Ze5hWTgqdCaXcHoIEZNtPkgWe9vEuvEvQO5gQ0UUoPgu0eUMu+GvZhXm1FfUUTeTu9MLnQC.AR4fS+ovPvv.GJtP18LjjYm3fJElBWW9YAwPAxdXO2a6.0kf0RaTx3iAI8NB5w2LYY1YOn+A9L42BaWyl63CVDPwrjAeh69d1+EvQu8UIerlk3QuoSd6ytuOSDczQikuhUh0r10gjSN4gX+q+gWe9vE9pKhibzi2c82stuiNO+ZPg1..MrghJpnTyNuBN9xVVQyXcqYkg78O.Jpu95vQOxQvYN8oPO85QHb1.vdt9o2JXBsuhHGHrGC3+BRFanx2ygNDMDAJqof9KMBC39U.4YnJ042hbZIJmSctG5CE56G4n+f7J6fpjWdLZ3A0wSPHz8bdIVTp7fh.61eXa2RjGwe66nhrv+SoO8okIV+ZWOV7RJBgGd3gf9oZTV4UfC9YG0a0W6F+G+jO3c9uMhwHMdjEZC.zvQrlMU77yLyIcvUtrkkxJVwRQTg30cLEc8vGhSc5Sgierihae6aa4Qthvs5j25JW1fb5YJYffQEBFMPdV+ampjv5xh7vPn9ABZHkKagvdStO0HApRy9KYF1LoQpf5FzzKyeIk2BKsSEJyGdgt2NrmCeKoUtx5sSGjRA.4ck.Oiv6fJTpq9D4C776GQXgiBKbQX0qcsH2bmdHqeqB0bsqiCe3iYVYkU+a0qmeM5OnM.PiADqe6aecYj7T20JKZIIt7kWDlPjg9kiDEWqlZvIOwwwYO6YPe80KDy4u5I4kgx5APPykrW+VvDlvU+FIhAl+gTvLLfj9.50sE8ffndDjUn2egeWYd3Uv+PWt4CFHJ4Vg3mOT0OAQPLm+17vWRYtUpBHrTA+kMT0zzDSaZYhUrxUikTTQHpnlXnni6Hps1ahCcjiYVZ4Urq8ti2p3QTlowiEPa.fFAM1xVJ94SIiTemUs7kF2xW5RPDQDwHJ+5t6tw4N2YwWbxSfqc8qId57Io8xtgAVPXhddKHpVSNetrmwQO9UveZjBDULMF8mYxJnUbOqu57yLxqLWEDG+jOBcoi2PoBeI5U3YtxP+KnjmwKVaaJ79ip00uoo+b6W3hVLVwpVExZZYEZGVTfabqagidziadoKWxGsuc81uLz6a+ZDjPa.fFCZ7hu32X6ImVpuwJV5hicoKcwg7sjTU3tM1HN8YNEN6oOCt+8um3MISBKrq.R8t2lRcFIRA91PbBdKiF5m+TgnLQNz2NthDfT5GFMhjv3XnxPI6i+lAttB5UEIGp28JLTvVN6o1VJ3kukT5j29Ftbg4MumBEszkh4MumZDM29Lb8ZuAN9I+Bbwu5JGbe69cdQ.3YDmoZ7XEdxcFGMF134douwykzjR7sVwxJZRKeoKdDsXAYvzzGpo5ZvoO8ovWcgyiG9vGZaiqgNouiG+tfryAJD0.wI8EB+Kff2gCz9S.idq7PSnmHapMDn+LHXr1XggB+EWJczwNgTtKoj113OraXkXE8KxU6duKVmHNcH9H+apXDDrdYH2bmNV7RJBEt3EOhrqZpBUTY03yO9IMqn7p9fOXWuUwP6wuFCQnM.PigM1xV9ZEEaJIsmkujBSakKeYH93iaTgud85EkUVY37e44vkuzEwCe3CsTtnHLtzW2c57HPrHBMIJkL3sCUQEToPCVdrxUUFfPYEUpxeO+5JiTgUOP1fFg9kiO6PApn2Y9i.xq3NnmnRUYk1pUfae7meGhQ.xskpp4mFIA6GjQD4WYMLH1+yNmbPgEtHrfBWDRZRSJHF+F9vqOenjRKCm7KNsupqo1e+d2wa98GUXrFOVCsA.ZDxvJKt3BROrn28BV3Sm+xW1RwTmZZiZ71ae8gRKqTbgyedbkKeIzYmcp74rpmN00L.od6.McAjVX.73W5Tfqeo2thUUd+Jl6aK4WdSmQTlUov1N+ECutySGXmWCD+Yz4PCFnS4L+odhqncjLvytQPVxhbTAbbW+iZOiDOxImbv7WPgnvEsHjTRI4PmJzCOd7fyegKhyb1y0as02vu7i1wa+eXTi4Z7XOzF.nQHG4latw+TEtrcVPAydCKeoKwH+YlWfkU2nC75yGtV0UiKdwuBW9xWBMc26Bgygf.vIuRkhSuPH6kOFhoPLOzhzSUN1+za8bLkXVxqr7KleaadkaXPB4tZ9SKxNKu0ESgh5c2tff+PjGjdJ+4TweAuxsonOHnm9LFv+gEjC0lg311qk.31saL6BlMd5mdAXdO0Si3iOdYgXDEs2d63Tm4r3be44a8tM21Od+67sd6QUAPimHf1..MFQwVK9a8+N6rx5GtjBWXXye9O0H9RHTEZng5wkuzkPIkbUTS0USVW3.hJXEiJffwAHfRDtKifPGjhjf57HyTrwNI.USu8knlfhZozMHFoA6gjm19NwegmK.482RiKX4u8vzKtyLZSwtjxdai+Rsi8N.HJxUYbm7V4q03jOXhDRHQL24LGL24NOL64L2wj2Uqqt5vYN2Ev49pKdsF6rkhO9d1yEF0EBMdhAZC.zXTAO+q7sesDiM5eUgKbAIUzhWHl7jm7XhbzUWcgxKuLTxUuBJsjRPKszLT8mAxEumPQmwTrP7tWNxBTkfVWChzG3a1THSiHtszA3fhURnxsuaFZI6p3uy2SJz+CB963dle+D5dQ9KO94TZ.BN5ofUzntc6F4kWdnf4LWLm4LWjdFYX6YGMPe80Gt7UtB9xyeIekUU0GnlRtv2nzRKsiwDgQimnf1..MFUQ9acqSsfID6NlcAErrhJbAFyZV4C2tcOlIO2ow6fJJubTYEkiJqnBzQGczugx2RourW4.p+yo.pMsst0E2DZrafg+u6TQMJ9cFOTkycwTA3D+AfCdl6Pn3GP9SiBfrwFhQHH34uD8DiNDoWNUDFbYNqokMlY94i7m0rvLxaFHhw.u7Y39szBN2Wddb9Kbw1a5tM9e5i1yN9kiYBiFOQBsA.ZLlgst8W6uOkIm7OcAKXdSbwKbgHojFcpnZmfooOzPCMfJJubTc0UipqoZzdqsBZX48qvxtRcwviK5UJET5oFUHGda4k4F6YXeW01gb+Ubgr6oj+LVQRSfSzG77WR9gTTUn5mEFfbf+RQcfNl4D8tb6FSaZSCSe54gYLiYfYle9i36FeCD5qu9PokUAtvWcQyRKszKdWSOe6SricT5XpPowSrPa.fFi4XiabiyH1Ik5aNqYOqkrvm9oMl6bl8nxFoRvf6cu6hZptFTSM0fpqoZbmFZ.974STAnbn5sE9do7XS9WXBX3xvthPJDLPfTFBTOe4JE6G9KDtdqBfS7yCD8AI+sI+1uNcOXP4lxiB9KCVexGLQrwDKxImbPt4NcjWdy.Ykc1i36VkAKZ7t2Em+BeEtvWc41a99s7eee69c96FqkIMzPa.fFiqvKT727uJwXS7u4oep4j3S+TyEYmUlipqffABd5oGbqacKT6MpE2n1qiaT6MvcuaiPN711BGurW9Dke9g8BmyBpSCAfTtwIZucNm4iFz6f7S6mB8ewBezV6J.+zOgILALsrlFxJqbPVYmMxJqrGwNRcGpnyN6.W5xkfqVRY9ppppNQms1z29fG7f2ZrVtzPCFzF.nw3RTPAEDS9y9o+0SI8z+ZO0bmSjye9yCoLNaBdF5t6tQC0WOtUc2B0Wecn95pC2ptagd6wi5vkqJL27by6+KxoCvVcBPR2f3JPvOrU07j5JXTgdExOuqZX.SSeP1n.JXEpmooIl7jmLlZ5YfLxHCjQFYhzyHcjbxIOtxvPF5s2dQYkWItxUtJtRIkWSGd67mef2+8+vwZ4RCMTAsA.ZLtGqYMaeVIL4I9axOubW9rlU9tl2bK.IlXhi0hU+BSSen4Vd.Z7N2FMdm6f6bmaiFuSi3N241nkVe...Y6K1RAnJOeUsLA8ecoP6KoT1+CYEhegC7lQP5kCUupzhHt8MahvBKbjRpohzlRZH0ojJlxTRCoNkzPZol5XZg5ELvqWunppqAkTR43JkWVys0bq+RcH903QAnM.PiGovl21VVdTgmv+X9yblEN6YMCWyatyEIjvn6lzxvEd73A2+d2CMcu6h60z8v8u+8v8t28PKszBdPKsf1ZqMAOp8ASXPL.X.CQuTkuKdsQZ5sGQ.evDQDV3HwIkHRH9DQRImDRJ4TPJImLRJ4jQJojBhKt3FW5QuSvqWun5ZtFtZokgRJo7GzZaO3026Ndm+8Puu7qwiPPa.fFOxhm8YKdMSH1H+EyLubWPd4kmqYWP9HsTScrVrF1vae8gGzZqnkVZFs9fVQ6s2JZqiNPms2NZu8NPGs2NZu81P2c2C5pqNQO8HdHvQCe9.s7DGnvu6D8tb6BQE0DwDiJJD0DmHhMlXQrwEKhIlXPrwFGhM1XQbwEGRHgDQBIlvnxAE0HM5tmdPkUUMprhZP4UTwCZu0V986xuRe8ovmFORBsA.Z7XA13F237hIwT+GlZ5Sc0ydlyHhYke9XZYkIb65QGuJGpvqOen6tdH5pqtQOc2M7zqGzqmdQu85Ad5sWzau8h95qO3ymOX5ym++0zG750DtbY.CCWvsaWvvv.tb4BFtbgvBObDY3giHhHRDdDgivBOBDQDQ3WgeTQMto55GoQ6s2NJsrJPkUUkY4kUU8Or2t9m9vc7G9uNVKWZnQn.ZC.z3wNjwl27jJL1j+kwmPBaM+YN8DxM2bP9yXFHt3hcrVzzXbN750Kt4MqCUVcUnlqUq2qUy0uZuFd961+N1wNGqkMMzHTCsA.Z7XO1x1+VemIDlwOOqbxcdSOmrBKuomCxZZSabydMfFis3dM2Lpo5qgqWasn7Jqt4tZu88U2Mq9u9Lm4LMNVKaZnwHIzF.nwSTXMqYMIG+jS+uMxHBea4kadSM6rmlwzyIGjYloOltkDqwnGZs01PMWqFb8ZuEtwMtd200PSmyrmd+G+vO7818XsrogFilPa.fFOQihVy1xdJIG8eazwF8ylSN4jRlSMMiLmVlHqok4XxoAmFgVXZ5CM0z8v0uwsP800.p6l00yMp+VWtOCe+y6eGu6qOVKeZnwXIzF.ngFDTTQEkZJSK6+5I3NxmOqryJmzm5TCKizSCYlQFH4jmziTKUsmDQWc2Mpu95wsp+1n95q2r1abi1dPqsdVuOru+YsG9ZngHzF.ngF8Ob+7E+pe2vc49UiN5nKLqzyLwzl5TLRepSEoO0zvjlTBZiBFiP28zCtysaD0Uec31M1HZng634F0UWM8AeG1SyM8q9rO6ypZrVF0PiwyPa.fFZLXwRWZTuPZY9m6JrH2ZrSL54lQlScRojRxtRMkTPZSYJH0TSAQEUTi0R4iMvqOenklaAMbm6fFa7t3d26d3N24tc2PC2tVev3Td6pm+oO5id+yMVKmZnwiZPa.fFZDhv5291WWrti96XBeKN4IkT1ok5jmXhIknqjlThXxojBRJ4jPhwGutXCc.OrqtPy2+93tMw1YDaEMc+60a82tw61yC6sbud68StTWM+6p6fGr4wZYUCMdb.ZC.zPiQXrlMU77iO1HdU31nH2gYjSZoLkTSXRIDYhIDuQ7wEOlThwi3hKNDW7wgDhKNDVXgMVKxiHn8N5.s2V6n01ZCO3AshVZsUzQashlePa80Xi2q8N6r0a0qOTpKO88G2ai238voOcWi0xrFZ73LzF.ngFignnhJJ0jmVNOuKutWrgaiY4BH8DmTBSNoDSL5nm3DcOwXh1HpnhBSLpnPzQOQDaLwfnln+uOgHm.lvDhbLY+LvqOeviGOn6t5Bc2cO3gc0E5ryGhN6rSzQmchG1U2n6tdH5ryG5q0VaySiMcuV7zSu21soYs8Bbo9dnmicfCriiMpK3ZngFbnM.PCMdD.adyaNyviO9hBCgMWSCW4h9LS0vERzvMhKL2QFWLwFczwNwXhH7Hc61s6HbEY3gY31saivCOLC2gGFbY3FtMLfogALB7eg4x.9LA7YZBSelvD9fOSS.ulnOSuvau8Z1qmdgWell80mGed7zmut6o695nyN5piV6rcetP6tAZwzmw8831aCnWuk1muNt3A26G+kPu+3qgFi6w+evab3XhRQXFR.....jTQNQjqBAlf" ],
													"embed" : 1,
													"id" : "obj-15",
													"maxclass" : "fpic",
													"numinlets" : 1,
													"numoutlets" : 1,
													"outlettype" : [ "jit_matrix" ],
													"patching_rect" : [ 24.0, 16.0, 32.0, 32.0 ],
													"presentation" : 1,
													"presentation_rect" : [ 18.0, 17.0, 44.0, 44.0 ]
												}

											}
, 											{
												"box" : 												{
													"fontname" : "Arial Bold",
													"fontsize" : 14.0,
													"id" : "obj-12",
													"maxclass" : "comment",
													"numinlets" : 1,
													"numoutlets" : 0,
													"patching_rect" : [ 24.0, 56.0, 196.0, 22.0 ],
													"presentation" : 1,
													"presentation_rect" : [ 71.0, 18.0, 196.0, 22.0 ],
													"text" : "This device requires Max 7!",
													"textcolor" : [ 0.258824, 0.258824, 0.258824, 1.0 ],
													"textjustification" : 1
												}

											}
, 											{
												"box" : 												{
													"angle" : 270.0,
													"bordercolor" : [ 0.490196, 0.482353, 0.478431, 1.0 ],
													"grad1" : [ 0.839216, 0.839216, 0.839216, 1.0 ],
													"grad2" : [ 0.473037, 0.473037, 0.473037, 1.0 ],
													"id" : "obj-3",
													"maxclass" : "panel",
													"mode" : 1,
													"numinlets" : 1,
													"numoutlets" : 0,
													"patching_rect" : [ 112.0, 16.0, 32.0, 32.0 ],
													"presentation" : 1,
													"presentation_rect" : [ 8.0, 8.0, 280.0, 80.0 ],
													"proportion" : 0.39
												}

											}
, 											{
												"box" : 												{
													"fontname" : "Arial Bold",
													"fontsize" : 10.0,
													"id" : "obj-98",
													"linecount" : 2,
													"maxclass" : "message",
													"numinlets" : 2,
													"numoutlets" : 1,
													"outlettype" : [ "" ],
													"patching_rect" : [ 24.0, 120.0, 320.0, 31.0 ],
													"text" : ";\rmax launchbrowser http://cycling74.com/downloads/max7forlive/"
												}

											}
, 											{
												"box" : 												{
													"angle" : 270.0,
													"bordercolor" : [ 0.094118, 0.113725, 0.137255, 1.0 ],
													"grad1" : [ 0.094118, 0.113725, 0.137255, 0.75 ],
													"grad2" : [ 0.094118, 0.113725, 0.137255, 0.75 ],
													"id" : "obj-73",
													"maxclass" : "panel",
													"mode" : 1,
													"numinlets" : 1,
													"numoutlets" : 0,
													"patching_rect" : [ 152.0, 16.0, 32.0, 32.0 ],
													"presentation" : 1,
													"presentation_rect" : [ 0.0, 0.0, 296.0, 96.0 ],
													"proportion" : 0.39,
													"prototypename" : "M4L.horizontal-black",
													"rounded" : 16
												}

											}
 ],
										"lines" : [ 											{
												"patchline" : 												{
													"destination" : [ "obj-98", 0 ],
													"source" : [ "obj-20", 0 ]
												}

											}
 ],
										"bgcolor" : [ 1.0, 1.0, 1.0, 0.0 ]
									}
,
									"patching_rect" : [ 216.0, 64.0, 297.0, 97.0 ],
									"presentation" : 1,
									"presentation_rect" : [ 136.0, 32.0, 297.0, 97.0 ],
									"viewvisibility" : 1
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 11.0,
									"id" : "obj-1",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 216.0, 32.0, 261.0, 19.0 ],
									"text" : "Display warning if Max version is smaller than 7.0.0"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-2",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 40.0, 400.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-16",
									"linecount" : 2,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 72.0, 88.0, 19.0, 29.0 ],
									"text" : "l\nl"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-23",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 104.0, 32.0, 83.0, 18.0 ],
									"text" : "Get Max version"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-27",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "bang" ],
									"patching_rect" : [ 40.0, 32.0, 55.0, 20.0 ],
									"text" : "loadbang"
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-32",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 40.0, 208.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-33",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 104.0, 160.0, 47.0, 18.0 ],
									"text" : "Decimal"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-34",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 104.0, 136.0, 68.0, 18.0 ],
									"text" : "Hexadecimal"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-35",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"patching_rect" : [ 40.0, 184.0, 47.0, 20.0 ],
									"text" : ">= 1792"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-36",
									"maxclass" : "number",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 40.0, 160.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 1,
									"id" : "obj-37",
									"maxclass" : "number",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 40.0, 136.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-38",
									"linecount" : 2,
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 40.0, 64.0, 152.0, 31.0 ],
									"text" : ";\rmax getversion ---MaxVersion"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-39",
									"maxclass" : "newobj",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 40.0, 112.0, 83.0, 20.0 ],
									"text" : "r ---MaxVersion"
								}

							}
, 							{
								"box" : 								{
									"angle" : 270.0,
									"bordercolor" : [ 0.537255, 0.537255, 0.537255, 1.0 ],
									"grad1" : [ 0.839216, 0.839216, 0.839216, 0.8 ],
									"grad2" : [ 0.266667, 0.266667, 0.266667, 0.8 ],
									"id" : "obj-71",
									"maxclass" : "panel",
									"mode" : 1,
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 218.0, 170.0, 296.0, 96.0 ],
									"presentation" : 1,
									"presentation_rect" : [ 0.0, 0.0, 571.0, 169.0 ],
									"proportion" : 0.39,
									"rounded" : 0
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-2", 0 ],
									"source" : [ "obj-10", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-38", 0 ],
									"source" : [ "obj-27", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"order" : 1,
									"source" : [ "obj-32", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-8", 0 ],
									"order" : 0,
									"source" : [ "obj-32", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-32", 0 ],
									"source" : [ "obj-35", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-35", 0 ],
									"source" : [ "obj-36", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-36", 0 ],
									"source" : [ "obj-37", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-37", 0 ],
									"source" : [ "obj-39", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-10", 0 ],
									"source" : [ "obj-4", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-7", 0 ],
									"source" : [ "obj-5", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-9", 0 ],
									"source" : [ "obj-5", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-4", 0 ],
									"source" : [ "obj-7", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-10", 0 ],
									"source" : [ "obj-8", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-4", 0 ],
									"source" : [ "obj-9", 0 ]
								}

							}
 ],
						"styles" : [ 							{
								"name" : "AudioStatus_Menu",
								"default" : 								{
									"bgfillcolor" : 									{
										"angle" : 270.0,
										"autogradient" : 0,
										"color" : [ 0.294118, 0.313726, 0.337255, 1 ],
										"color1" : [ 0.454902, 0.462745, 0.482353, 0.0 ],
										"color2" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
										"proportion" : 0.39,
										"type" : "color"
									}

								}
,
								"parentstyle" : "",
								"multi" : 0
							}
 ],
						"bgcolor" : [ 1.0, 1.0, 1.0, 0.0 ]
					}
,
					"patching_rect" : [ 358.064518690109253, 1166.129040598869324, 16.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 0.0, 1.0, 10.0, 10.0 ],
					"varname" : "Max6Warning",
					"viewvisibility" : 1
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-98",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 358.064518690109253, 1190.322589159011841, 64.0, 20.0 ],
					"save" : [ "#N", "thispatcher", ";", "#Q", "end", ";" ],
					"text" : "thispatcher"
				}

			}
, 			{
				"box" : 				{
					"angle" : 0.0,
					"bgcolor" : [ 0.0, 0.0, 0.0, 1.0 ],
					"id" : "obj-31",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 788.709683060646057, 1190.322589159011841, 16.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 326.0, 24.0, 6.0, 76.0 ],
					"proportion" : 0.39,
					"prototypename" : "M4L.horizontal-black",
					"rounded" : 0
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-58",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 558.064520120620728, 719.354843854904175, 33.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 19.0, 135.0, 33.0, 18.0 ],
					"text" : "Fade"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-54",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 269.354840636253357, 951.612910032272339, 65.0, 18.0 ],
					"text" : "< MIDI notes"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-17",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 550.000003933906555, 759.677424788475037, 48.0, 20.0 ],
					"text" : "Fade $1"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Disables all held notes.",
					"automation" : "arm",
					"automationon" : "trig",
					"id" : "obj-19",
					"maxclass" : "live.text",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 341.935486316680908, 854.838715791702271, 36.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 526.0, 144.0, 34.0, 15.0 ],
					"prototypename" : "numbers.default",
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "arm", "trig" ],
							"parameter_invisible" : 2,
							"parameter_linknames" : 1,
							"parameter_longname" : "Stop",
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Stop",
							"parameter_type" : 2
						}

					}
,
					"text" : "Stop",
					"texton" : "Stop",
					"varname" : "Stop"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Enables/disables reversed playback.",
					"appearance" : 1,
					"automation" : "off",
					"automationon" : "on",
					"id" : "obj-73",
					"maxclass" : "live.text",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 974.193555355072021, 951.612910032272339, 48.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 160.0, 153.0, 45.0, 14.0 ],
					"rounded" : 64.0,
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Reverse",
							"parameter_mapping_index" : 19,
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Reverse",
							"parameter_type" : 2
						}

					}
,
					"text" : "Rev",
					"texton" : "Rev",
					"varname" : "Reverse"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-85",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 541.935487747192383, 951.612910032272339, 65.0, 18.0 ],
					"text" : "Messages >"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-79",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 637.096778750419617, 998.387103915214539, 24.0, 18.0 ],
					"text" : "R~"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-78",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 230.64516294002533, 998.387103915214539, 22.0, 18.0 ],
					"text" : "L~"
				}

			}
, 			{
				"box" : 				{
					"arrows" : 1,
					"border" : 0.5,
					"id" : "obj-76",
					"linecolor" : [ 0.0, 0.0, 0.0, 1.0 ],
					"maxclass" : "live.line",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 396.77419638633728, 998.387103915214539, 8.0, 40.0 ],
					"saved_attribute_attributes" : 					{
						"linecolor" : 						{
							"expression" : ""
						}

					}

				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-74",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 374.193551063537598, 951.612910032272339, 129.0, 18.0 ],
					"text" : "Polyphonic sampler DSP"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-71",
					"linecount" : 7,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 358.064518690109253, 1038.7096848487854, 184.0, 85.0 ],
					"text" : "Poly~ container:\nArgument: patcher name\nAttributes:\n@voices 10: nb of voices\n@steal 1: voice stealing\n@args: buffer name\n@target 0: send messages to all voices"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-67",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 293.548389196395874, 854.838715791702271, 33.0, 18.0 ],
					"text" : "Pitch"
				}

			}
, 			{
				"box" : 				{
					"color" : [ 1.0, 0.788235, 0.027451, 1.0 ],
					"fontface" : 0,
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-55",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "signal", "signal" ],
					"patching_rect" : [ 252.90322732925415, 961.0, 413.0, 20.0 ],
					"text" : "poly~ M4L.SimplerPolyVoice~ @voices 10 @steal 1 @target 0 @args ---samplerbuffer"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-40",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "signal", "signal" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 6,
							"revision" : 5,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 109.0, 153.0, 236.0, 326.0 ],
						"bglocked" : 0,
						"openinpresentation" : 0,
						"default_fontsize" : 10.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial Bold",
						"gridonopen" : 1,
						"gridsize" : [ 8.0, 8.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"boxes" : [ 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-10",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 84.0, 140.0, 63.0, 18.0 ],
									"text" : "Sample info"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 11.0,
									"id" : "obj-57",
									"linecount" : 3,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 40.0, 16.0, 160.0, 43.0 ],
									"text" : "If sample is stereo, play L & R channels, if it's mono, play L channel in both outputs."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-9",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 1,
									"outlettype" : [ "signal" ],
									"patching_rect" : [ 88.0, 208.0, 67.0, 20.0 ],
									"text" : "selector~ 2"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-26",
									"maxclass" : "number",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 88.0, 184.0, 46.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-25",
									"maxclass" : "newobj",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 88.0, 160.0, 108.0, 20.0 ],
									"text" : "r ---SampleChannels"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 11.0,
									"id" : "obj-3",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 136.0, 72.0, 23.0, 19.0 ],
									"text" : "R~"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 11.0,
									"id" : "obj-4",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 40.0, 72.0, 25.0, 19.0 ],
									"text" : "L~"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-2",
									"index" : 2,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "signal" ],
									"patching_rect" : [ 136.0, 96.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-1",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "signal" ],
									"patching_rect" : [ 40.0, 96.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 11.0,
									"id" : "obj-21",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 88.0, 280.0, 23.0, 19.0 ],
									"text" : "R~"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 11.0,
									"id" : "obj-20",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 40.0, 280.0, 25.0, 19.0 ],
									"text" : "L~"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-141",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 40.0, 248.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-142",
									"index" : 2,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 88.0, 248.0, 25.0, 25.0 ]
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-141", 0 ],
									"order" : 1,
									"source" : [ "obj-1", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-9", 1 ],
									"order" : 0,
									"source" : [ "obj-1", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-9", 2 ],
									"source" : [ "obj-2", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-26", 0 ],
									"source" : [ "obj-25", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-9", 0 ],
									"source" : [ "obj-26", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-142", 0 ],
									"source" : [ "obj-9", 0 ]
								}

							}
 ]
					}
,
					"patching_rect" : [ 245.16129207611084, 998.387103915214539, 389.089109480381012, 20.0 ],
					"saved_object_attributes" : 					{
						"description" : "",
						"digest" : "",
						"fontname" : "Arial Bold",
						"fontsize" : 10.0,
						"globalpatchername" : "",
						"tags" : ""
					}
,
					"text" : "p Channels~"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-21",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 685.483875870704651, 998.387103915214539, 100.0, 20.0 ],
					"text" : "prepend PitchTime"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-64",
					"linecolor" : [ 0.4, 0.4, 0.4, 1.0 ],
					"maxclass" : "live.line",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 861.290328741073608, 1190.322589159011841, 8.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 425.0, 101.0, 50.0, 5.0 ],
					"saved_attribute_attributes" : 					{
						"linecolor" : 						{
							"expression" : ""
						}

					}

				}

			}
, 			{
				"box" : 				{
					"id" : "obj-60",
					"linecolor" : [ 0.4, 0.4, 0.4, 1.0 ],
					"maxclass" : "live.line",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 845.161296367645264, 1190.322589159011841, 8.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 425.0, 101.0, 5.0, 11.0 ],
					"saved_attribute_attributes" : 					{
						"linecolor" : 						{
							"expression" : ""
						}

					}

				}

			}
, 			{
				"box" : 				{
					"id" : "obj-13",
					"linecolor" : [ 0.4, 0.4, 0.4, 1.0 ],
					"maxclass" : "live.line",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 869.354844927787781, 1190.322589159011841, 8.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 481.0, 109.0, 5.0, 5.0 ],
					"saved_attribute_attributes" : 					{
						"linecolor" : 						{
							"expression" : ""
						}

					}

				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"linecolor" : [ 0.4, 0.4, 0.4, 1.0 ],
					"maxclass" : "live.line",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 838.709683418273926, 1190.322589159011841, 8.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 481.0, 89.0, 5.0, 6.0 ],
					"saved_attribute_attributes" : 					{
						"linecolor" : 						{
							"expression" : ""
						}

					}

				}

			}
, 			{
				"box" : 				{
					"id" : "obj-48",
					"maxclass" : "live.line",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 853.225812554359436, 1190.322589159011841, 8.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 392.0, 8.0, 8.0, 152.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-44",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 812.903231620788574, 838.709683418273926, 27.5, 20.0 ],
					"text" : "init"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Resets stretch, transp and formant parameters.",
					"automation" : "arm",
					"automationon" : "trig",
					"fontsize" : 8.0,
					"id" : "obj-28",
					"maxclass" : "live.text",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 812.903231620788574, 814.516134858131409, 24.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 475.0, 95.0, 14.0, 14.0 ],
					"prototypename" : "numbers.default",
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "arm", "trig" ],
							"parameter_invisible" : 2,
							"parameter_linknames" : 1,
							"parameter_longname" : "ResetPitchTime",
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Reset",
							"parameter_type" : 2
						}

					}
,
					"text" : "R",
					"texton" : "R",
					"varname" : "ResetPitchTime"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-22",
					"maxclass" : "live.line",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 830.645167231559753, 1190.322589159011841, 8.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 516.0, 8.0, 8.0, 152.0 ]
				}

			}
, 			{
				"box" : 				{
					"activetextcolor" : [ 0.0, 0.0, 0.0, 0.0 ],
					"activetextoncolor" : [ 0.0, 0.0, 0.0, 0.0 ],
					"annotation" : "Enables/disables time stretching.",
					"appearance" : 1,
					"automation" : "off",
					"automationon" : "on",
					"id" : "obj-57",
					"maxclass" : "live.text",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 717.74194061756134, 951.612910032272339, 24.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 497.0, 63.0, 16.0, 12.0 ],
					"rounded" : 16.0,
					"saved_attribute_attributes" : 					{
						"activetextcolor" : 						{
							"expression" : ""
						}
,
						"activetextoncolor" : 						{
							"expression" : ""
						}
,
						"textcolor" : 						{
							"expression" : ""
						}
,
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_initial" : [ 1 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "EnableStretch",
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Enable",
							"parameter_type" : 2
						}

					}
,
					"text" : "Off",
					"textcolor" : [ 0.321569, 0.321569, 0.321569, 0.0 ],
					"texton" : "On",
					"varname" : "EnableStretch"
				}

			}
, 			{
				"box" : 				{
					"activetextcolor" : [ 0.0, 0.0, 0.0, 0.0 ],
					"activetextoncolor" : [ 0.0, 0.0, 0.0, 0.0 ],
					"annotation" : "Enables/disables formant transposition.",
					"appearance" : 1,
					"automation" : "off",
					"automationon" : "on",
					"id" : "obj-53",
					"maxclass" : "live.text",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 845.161296367645264, 951.612910032272339, 24.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 497.0, 138.0, 16.0, 12.0 ],
					"rounded" : 16.0,
					"saved_attribute_attributes" : 					{
						"activetextcolor" : 						{
							"expression" : ""
						}
,
						"activetextoncolor" : 						{
							"expression" : ""
						}
,
						"textcolor" : 						{
							"expression" : ""
						}
,
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_initial" : [ 1 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "EnableFormant",
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Enable",
							"parameter_type" : 2
						}

					}
,
					"text" : "Off",
					"textcolor" : [ 0.321569, 0.321569, 0.321569, 0.0 ],
					"texton" : "On",
					"varname" : "EnableFormant"
				}

			}
, 			{
				"box" : 				{
					"activetextcolor" : [ 0.0, 0.0, 0.0, 0.0 ],
					"activetextoncolor" : [ 0.0, 0.0, 0.0, 0.0 ],
					"annotation" : "Enables/disables transposition.",
					"appearance" : 1,
					"automation" : "off",
					"automationon" : "on",
					"id" : "obj-49",
					"maxclass" : "live.text",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 782.258070111274719, 951.612910032272339, 24.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 441.0, 138.0, 16.0, 12.0 ],
					"rounded" : 16.0,
					"saved_attribute_attributes" : 					{
						"activetextcolor" : 						{
							"expression" : ""
						}
,
						"activetextoncolor" : 						{
							"expression" : ""
						}
,
						"textcolor" : 						{
							"expression" : ""
						}
,
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_initial" : [ 1 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "EnableTransp",
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Enable",
							"parameter_type" : 2
						}

					}
,
					"text" : "Off",
					"textcolor" : [ 0.321569, 0.321569, 0.321569, 0.0 ],
					"texton" : "On",
					"varname" : "EnableTransp"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-47",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 909.677425861358643, 935.483877658843994, 42.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 409.0, 58.0, 42.0, 18.0 ],
					"text" : "Quality"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-46",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 958.064522981643677, 895.161296725273132, 36.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 411.0, 26.0, 36.0, 18.0 ],
					"text" : "Mode"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sets time/pitch quality. \"Best\" quality is only available for \"basic\" and \"efficient\" modes.  In other modes, whenever \"best\" quality is selected, it is re-set to \"better\".",
					"id" : "obj-43",
					"maxclass" : "live.menu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 909.677425861358643, 951.612910032272339, 56.0, 15.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 401.0, 74.0, 56.0, 15.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "basic", "good", "better", "best" ],
							"parameter_linknames" : 1,
							"parameter_longname" : "Quality",
							"parameter_mapping_index" : 5,
							"parameter_mmax" : 3,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Quality",
							"parameter_type" : 2
						}

					}
,
					"varname" : "Quality"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sets time/pitch mode.",
					"id" : "obj-42",
					"maxclass" : "live.menu",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 941.935490608215332, 911.290329098701477, 64.0, 15.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 401.0, 42.0, 56.0, 15.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "basic", "mono", "rhythm", "general", "extreme", "efficient" ],
							"parameter_linknames" : 1,
							"parameter_longname" : "Mode",
							"parameter_mapping_index" : 4,
							"parameter_mmax" : 5,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Mode",
							"parameter_type" : 2
						}

					}
,
					"varname" : "Mode"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sets time stretching factor.",
					"id" : "obj-38",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 750.00000536441803, 879.032264351844788, 56.0, 54.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 460.0, 38.0, 44.0, 54.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_exponent" : 3.5,
							"parameter_initial" : [ 1.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Stretch",
							"parameter_mapping_index" : 1,
							"parameter_mmax" : 10.0,
							"parameter_mmin" : 0.1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Stretch",
							"parameter_type" : 0,
							"parameter_units" : "x",
							"parameter_unitstyle" : 9
						}

					}
,
					"triangle" : 1,
					"varname" : "Stretch"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-23",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 677.419359683990479, 903.225812911987305, 61.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 400.0, 8.0, 61.0, 18.0 ],
					"text" : "Pitch/Time"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Enables/disables pitch & time parameters.",
					"automation" : "off",
					"automationon" : "on",
					"id" : "obj-24",
					"maxclass" : "live.text",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 685.483875870704651, 925.806458234786987, 32.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 465.0, 8.0, 36.0, 16.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_initial" : [ 1 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "PitchTime",
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Enable",
							"parameter_type" : 2
						}

					}
,
					"text" : "Off",
					"texton" : "On",
					"varname" : "PitchTime"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-25",
					"maxclass" : "newobj",
					"numinlets" : 11,
					"numoutlets" : 5,
					"outlettype" : [ "", "", "", "", "" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 6,
							"revision" : 5,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 84.0, 144.0, 1061.0, 611.0 ],
						"bglocked" : 0,
						"openinpresentation" : 0,
						"default_fontsize" : 10.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial Bold",
						"gridonopen" : 1,
						"gridsize" : [ 8.0, 8.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"boxes" : [ 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-78",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 920.0, 296.0, 61.0, 20.0 ],
									"text" : "rootkey $1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-74",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 848.0, 296.0, 60.0, 20.0 ],
									"text" : "reverse $1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-59",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 912.0, 24.0, 45.0, 18.0 ],
									"text" : "Root"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-60",
									"index" : 11,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 920.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-73",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 920.0, 88.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-56",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 840.0, 24.0, 49.0, 18.0 ],
									"text" : "Reverse"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-57",
									"index" : 10,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 848.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-58",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 848.0, 88.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-104",
									"linecount" : 4,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 680.0, 416.0, 176.0, 51.0 ],
									"text" : "\"Best\" quality is only available for \"basic\" and \"efficient\" modes. \nFor other modes, when \"best\" quality is selected, it is re-set to \"better\"."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-103",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 704.0, 256.0, 52.0, 20.0 ],
									"text" : "deferlow"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-96",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 736.0, 296.0, 44.0, 18.0 ],
									"text" : "Quality"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-97",
									"index" : 5,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 704.0, 288.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-93",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "int", "int" ],
									"patching_rect" : [ 704.0, 136.0, 51.0, 20.0 ],
									"text" : "unpack"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-92",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 704.0, 112.0, 32.5, 20.0 ],
									"text" : "pak"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-87",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 2,
									"outlettype" : [ "", "" ],
									"patching_rect" : [ 704.0, 232.0, 75.0, 20.0 ],
									"text" : "substitute 3 2"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-83",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 2,
									"outlettype" : [ "", "" ],
									"patching_rect" : [ 680.0, 208.0, 43.0, 20.0 ],
									"text" : "gate 2"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-75",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 2,
									"outlettype" : [ "", "" ],
									"patching_rect" : [ 736.0, 160.0, 104.0, 20.0 ],
									"text" : "zl lookup 1 2 2 2 2 1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-72",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "bang", "int" ],
									"patching_rect" : [ 152.0, 112.0, 33.5, 20.0 ],
									"text" : "t b i"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-1",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 296.0, 154.0, 32.5, 20.0 ],
									"text" : "* 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-69",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 296.0, 130.0, 51.0, 20.0 ],
									"text" : "pak 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-71",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 296.0, 296.0, 53.0, 20.0 ],
									"text" : "speed $1"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-70",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 48.0, 520.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-47",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 296.0, 242.0, 32.5, 20.0 ],
									"text" : "!/ 1."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-50",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 344.0, 272.0, 39.0, 18.0 ],
									"text" : "Speed"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-68",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 296.0, 272.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-102",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 344.0, 224.0, 36.0, 18.0 ],
									"text" : "factor"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-100",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 296.0, 214.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-101",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 2,
									"outlettype" : [ "", "" ],
									"patching_rect" : [ 296.0, 190.0, 81.0, 20.0 ],
									"text" : "substitute 0. 1."
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-67",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 152.0, 88.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-62",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 544.0, 528.0, 50.0, 18.0 ],
									"text" : "Formant"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-63",
									"index" : 4,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 512.0, 520.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-64",
									"linecount" : 2,
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 512.0, 472.0, 72.0, 31.0 ],
									"text" : "active $1, outputvalue"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-65",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"patching_rect" : [ 512.0, 448.0, 32.5, 20.0 ],
									"text" : "*"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-66",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 512.0, 424.0, 34.5, 20.0 ],
									"text" : "pak"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-54",
									"linecount" : 2,
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 168.0, 472.0, 72.0, 31.0 ],
									"text" : "active $1, outputvalue"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-53",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 288.0, 528.0, 45.0, 18.0 ],
									"text" : "Stretch"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-51",
									"index" : 3,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 256.0, 520.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-45",
									"linecount" : 2,
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 256.0, 472.0, 72.0, 31.0 ],
									"text" : "active $1, outputvalue"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-26",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"patching_rect" : [ 256.0, 448.0, 32.5, 20.0 ],
									"text" : "*"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-19",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 256.0, 424.0, 34.5, 20.0 ],
									"text" : "pak"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-11",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 264.0, 24.0, 40.0, 18.0 ],
									"text" : "On/off"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-10",
									"index" : 2,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 272.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-3",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 272.0, 88.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-77",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 608.0, 72.0, 19.0, 18.0 ],
									"text" : "ct"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-76",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 480.0, 72.0, 19.0, 18.0 ],
									"text" : "ct"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-61",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 196.0, 528.0, 48.0, 18.0 ],
									"text" : "Other UI"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-55",
									"index" : 2,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 168.0, 520.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 11.0,
									"id" : "obj-52",
									"linecount" : 4,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 80.0, 496.0, 76.0, 56.0 ],
									"text" : "Some typical pitch/time messages to groove~"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-48",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 320.0, 24.0, 45.0, 18.0 ],
									"text" : "Stretch"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-49",
									"index" : 3,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 328.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-46",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 328.0, 88.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-44",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 552.0, 296.0, 62.0, 20.0 ],
									"text" : "formant $1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-43",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 600.0, 232.0, 36.0, 18.0 ],
									"text" : "factor"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-42",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 552.0, 232.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-41",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 552.0, 208.0, 112.0, 20.0 ],
									"text" : "expr pow(2.\\,$f1/1200)"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-31",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 584.0, 88.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-32",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 576.0, 24.0, 50.0, 18.0 ],
									"text" : "Formant"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-33",
									"index" : 7,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 584.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-34",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 600.0, 184.0, 19.0, 18.0 ],
									"text" : "ct"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-35",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 552.0, 184.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-36",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 552.0, 160.0, 32.5, 20.0 ],
									"text" : "* 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-37",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 552.0, 136.0, 51.0, 20.0 ],
									"text" : "pak 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-38",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 520.0, 24.0, 40.0, 18.0 ],
									"text" : "On/off"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-39",
									"index" : 6,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 528.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-40",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 528.0, 88.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-30",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 456.0, 88.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-28",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 448.0, 24.0, 44.0, 18.0 ],
									"text" : "Transp"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-29",
									"index" : 5,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 456.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-27",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 472.0, 184.0, 19.0, 18.0 ],
									"text" : "ct"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-25",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 424.0, 184.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-24",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 424.0, 160.0, 32.5, 20.0 ],
									"text" : "* 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-23",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 424.0, 136.0, 51.0, 20.0 ],
									"text" : "pak 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-21",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 392.0, 24.0, 40.0, 18.0 ],
									"text" : "On/off"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-22",
									"index" : 4,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 400.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-20",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 400.0, 88.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-18",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 424.0, 296.0, 55.0, 20.0 ],
									"text" : "transp $1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-14",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 696.0, 24.0, 44.0, 18.0 ],
									"text" : "Quality"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-15",
									"index" : 8,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 704.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-16",
									"items" : [ "basic", ",", "good", ",", "better", ",", "best" ],
									"maxclass" : "umenu",
									"numinlets" : 1,
									"numoutlets" : 3,
									"outlettype" : [ "int", "", "" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 680.0, 360.0, 67.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-17",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 704.0, 384.0, 57.0, 20.0 ],
									"text" : "quality $1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-13",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 776.0, 24.0, 37.0, 18.0 ],
									"text" : "Mode"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-12",
									"index" : 9,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 784.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-9",
									"items" : [ "basic", ",", "monophonic", ",", "rhythmic", ",", "general", ",", "extremestretch", ",", "efficient" ],
									"maxclass" : "umenu",
									"numinlets" : 1,
									"numoutlets" : 3,
									"outlettype" : [ "int", "", "" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 784.0, 360.0, 67.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-7",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 808.0, 384.0, 50.0, 20.0 ],
									"text" : "mode $1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-8",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 144.0, 24.0, 40.0, 18.0 ],
									"text" : "On/off"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-6",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 152.0, 48.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-5",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 424.0, 360.0, 19.0, 20.0 ],
									"text" : "t l"
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-4",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 168.0, 424.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-2",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 192.0, 296.0, 76.0, 20.0 ],
									"text" : "timestretch $1"
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-101", 0 ],
									"source" : [ "obj-1", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-3", 0 ],
									"source" : [ "obj-10", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-47", 0 ],
									"source" : [ "obj-100", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-100", 0 ],
									"source" : [ "obj-101", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-100", 0 ],
									"source" : [ "obj-101", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-97", 0 ],
									"source" : [ "obj-103", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-9", 0 ],
									"order" : 0,
									"source" : [ "obj-12", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-92", 1 ],
									"order" : 1,
									"source" : [ "obj-12", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-92", 0 ],
									"source" : [ "obj-15", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-17", 0 ],
									"source" : [ "obj-16", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-17", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-18", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-26", 0 ],
									"source" : [ "obj-19", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-2", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-23", 0 ],
									"source" : [ "obj-20", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-20", 0 ],
									"source" : [ "obj-22", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-24", 0 ],
									"source" : [ "obj-23", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-25", 0 ],
									"source" : [ "obj-24", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-18", 0 ],
									"source" : [ "obj-25", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-45", 0 ],
									"source" : [ "obj-26", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-30", 0 ],
									"source" : [ "obj-29", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-19", 1 ],
									"order" : 1,
									"source" : [ "obj-3", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-69", 0 ],
									"order" : 0,
									"source" : [ "obj-3", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-23", 1 ],
									"source" : [ "obj-30", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-37", 1 ],
									"source" : [ "obj-31", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-31", 0 ],
									"source" : [ "obj-33", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-41", 0 ],
									"source" : [ "obj-35", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-35", 0 ],
									"source" : [ "obj-36", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-36", 0 ],
									"source" : [ "obj-37", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-40", 0 ],
									"source" : [ "obj-39", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-19", 0 ],
									"order" : 1,
									"source" : [ "obj-4", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-54", 0 ],
									"order" : 2,
									"source" : [ "obj-4", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-66", 0 ],
									"order" : 0,
									"source" : [ "obj-4", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-37", 0 ],
									"order" : 0,
									"source" : [ "obj-40", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-66", 1 ],
									"order" : 1,
									"source" : [ "obj-40", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-42", 0 ],
									"source" : [ "obj-41", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-44", 0 ],
									"source" : [ "obj-42", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-44", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-51", 0 ],
									"source" : [ "obj-45", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-69", 1 ],
									"source" : [ "obj-46", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-68", 0 ],
									"source" : [ "obj-47", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-46", 0 ],
									"source" : [ "obj-49", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-70", 0 ],
									"midpoints" : [ 433.5, 391.5, 57.5, 391.5 ],
									"source" : [ "obj-5", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-55", 0 ],
									"source" : [ "obj-54", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-58", 0 ],
									"source" : [ "obj-57", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-74", 0 ],
									"source" : [ "obj-58", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-67", 0 ],
									"source" : [ "obj-6", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-73", 0 ],
									"source" : [ "obj-60", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-63", 0 ],
									"source" : [ "obj-64", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-64", 0 ],
									"source" : [ "obj-65", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-65", 0 ],
									"source" : [ "obj-66", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-72", 0 ],
									"source" : [ "obj-67", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-71", 0 ],
									"source" : [ "obj-68", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-1", 0 ],
									"source" : [ "obj-69", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-7", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-71", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-2", 0 ],
									"order" : 0,
									"source" : [ "obj-72", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-23", 0 ],
									"order" : 0,
									"source" : [ "obj-72", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-4", 0 ],
									"order" : 1,
									"source" : [ "obj-72", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-69", 0 ],
									"order" : 1,
									"source" : [ "obj-72", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-78", 0 ],
									"source" : [ "obj-73", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-74", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-83", 0 ],
									"source" : [ "obj-75", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-78", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-16", 0 ],
									"source" : [ "obj-83", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-87", 0 ],
									"source" : [ "obj-83", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-103", 0 ],
									"source" : [ "obj-87", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-16", 0 ],
									"source" : [ "obj-87", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-7", 0 ],
									"source" : [ "obj-9", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-93", 0 ],
									"source" : [ "obj-92", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-75", 0 ],
									"source" : [ "obj-93", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-83", 1 ],
									"source" : [ "obj-93", 0 ]
								}

							}
 ]
					}
,
					"patching_rect" : [ 685.483875870704651, 975.806458592414856, 338.5, 20.0 ],
					"saved_object_attributes" : 					{
						"description" : "",
						"digest" : "",
						"fontname" : "Arial Bold",
						"fontsize" : 10.0,
						"globalpatchername" : "",
						"tags" : ""
					}
,
					"text" : "p Pitch/Time"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontname" : "Arial",
					"fontsize" : 11.0,
					"id" : "obj-37",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 290.322582721710205, 1416.129042387008667, 82.0, 19.0 ],
					"text" : "Audio output"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sample's output gain.",
					"clip_size" : 1,
					"display_range" : [ -70.0, 30.0 ],
					"id" : "obj-45",
					"lastchannelcount" : 0,
					"maxclass" : "live.gain~",
					"numinlets" : 2,
					"numoutlets" : 5,
					"outlettype" : [ "signal", "signal", "", "float", "list" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 296.774195671081543, 1459.677429795265198, 70.0, 103.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 520.0, 11.0, 48.0, 125.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Gain",
							"parameter_mapping_index" : 8,
							"parameter_mmax" : 30.0,
							"parameter_mmin" : -70.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Gain",
							"parameter_type" : 0,
							"parameter_unitstyle" : 4
						}

					}
,
					"varname" : "Gain"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sets formant transposition in cents.",
					"id" : "obj-52",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 877.419361114501953, 879.032264351844788, 44.0, 54.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 454.0, 113.0, 56.0, 54.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Formant",
							"parameter_mapping_index" : 3,
							"parameter_mmax" : 1200.0,
							"parameter_mmin" : -1200.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Formant",
							"parameter_steps" : 481,
							"parameter_type" : 0,
							"parameter_units" : "ct",
							"parameter_unitstyle" : 9
						}

					}
,
					"triangle" : 1,
					"varname" : "Formant"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sets transposition in cents.",
					"id" : "obj-50",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 812.903231620788574, 879.032264351844788, 56.0, 54.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 398.0, 113.0, 56.0, 54.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Transp",
							"parameter_mapping_index" : 2,
							"parameter_mmax" : 2400.0,
							"parameter_mmin" : -2400.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Transp",
							"parameter_steps" : 961,
							"parameter_type" : 0,
							"parameter_units" : "ct",
							"parameter_unitstyle" : 9
						}

					}
,
					"triangle" : 1,
					"varname" : "Transp"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-198",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 333.870970129966736, 759.677424788475037, 68.0, 20.0 ],
					"text" : "VelSense $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-197",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 330.645163655281067, 714.516134142875671, 90.772278189659119, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 347.0, 32.0, 35.0, 18.0 ],
					"text" : "Sens"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Velocity sensitivity.",
					"appearance" : 2,
					"id" : "obj-196",
					"maxclass" : "live.numbox",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 333.870970129966736, 735.48387622833252, 40.0, 15.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 344.0, 47.0, 40.0, 15.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 50 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Sense",
							"parameter_mapping_index" : 11,
							"parameter_mmax" : 100.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Sense",
							"parameter_type" : 0,
							"parameter_unitstyle" : 5
						}

					}
,
					"varname" : "Sense"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-194",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1012.903233051300049, 919.354845285415649, 33.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 348.0, 2.0, 33.0, 18.0 ],
					"text" : "Root"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Original pitch.",
					"appearance" : 1,
					"id" : "obj-192",
					"maxclass" : "live.numbox",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 1004.838716864585876, 935.483877658843994, 40.0, 15.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 344.0, 17.0, 40.0, 15.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 60 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Root",
							"parameter_mapping_index" : 10,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Root",
							"parameter_type" : 0,
							"parameter_unitstyle" : 8
						}

					}
,
					"varname" : "Root"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-65",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 685.483875870704651, 790.322586297988892, 81.0, 20.0 ],
					"text" : "prepend ADSR"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-63",
					"maxclass" : "newobj",
					"numinlets" : 8,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 6,
							"revision" : 5,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 84.0, 144.0, 606.0, 697.0 ],
						"bglocked" : 0,
						"openinpresentation" : 0,
						"default_fontsize" : 10.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial Bold",
						"gridonopen" : 1,
						"gridsize" : [ 8.0, 8.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"boxes" : [ 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-22",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 264.0, 560.0, 275.0, 20.0 ],
									"text" : "clear, xyc 0. 0. 0., xyc $6 1. $7, xyc $3 $4 $5, xyc $1 0. $2"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-23",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 264.0, 532.0, 179.0, 20.0 ],
									"text" : "join 3"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-26",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 506.0, 344.0, 37.0, 18.0 ],
									"text" : "Curve"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-27",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 450.0, 344.0, 23.0, 18.0 ],
									"text" : "Att"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-28",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 392.0, 344.0, 28.0, 18.0 ],
									"text" : "Sus"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-29",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 338.0, 344.0, 28.0, 18.0 ],
									"text" : "Dec"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-30",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 280.0, 344.0, 26.0, 18.0 ],
									"text" : "Rel"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-6",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 504.0, 376.0, 39.0, 18.0 ],
									"text" : "[-1. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-7",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 488.0, 360.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-8",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 448.0, 376.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-9",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 432.0, 360.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-10",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 392.0, 376.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-11",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 376.0, 360.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-12",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 336.0, 376.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-14",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 320.0, 360.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-16",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 280.0, 376.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-17",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 264.0, 360.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-237",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 264.0, 480.0, 32.5, 20.0 ],
									"text" : "/ 3."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-233",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 344.0, 480.0, 32.5, 20.0 ],
									"text" : "/ 3."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-221",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 488.0, 408.0, 32.5, 20.0 ],
									"text" : "* -1."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-218",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 264.0, 456.0, 32.5, 20.0 ],
									"text" : "+ 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-217",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 344.0, 408.0, 32.5, 20.0 ],
									"text" : "+ 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-214",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 432.0, 504.0, 75.0, 20.0 ],
									"text" : "pack 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-213",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 264.0, 504.0, 72.0, 20.0 ],
									"text" : "pack 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-204",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 344.0, 504.0, 82.0, 20.0 ],
									"text" : "pack 0. 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-197",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 432.0, 480.0, 32.5, 20.0 ],
									"text" : "/ 3."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-196",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 5,
									"outlettype" : [ "float", "float", "float", "float", "float" ],
									"patching_rect" : [ 264.0, 320.0, 243.0, 20.0 ],
									"text" : "unpack 0. 0. 0. 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-74",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 88.0, 304.0, 100.0, 18.0 ],
									"text" : "Envelope for curve~"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-73",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 448.0, 216.0, 32.5, 20.0 ],
									"text" : "* -1."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-149",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 56.0, 320.0, 161.0, 20.0 ],
									"text" : "0. 0. 0. 1. $1 $6 $3 $2 $5 0. $4 $5"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-103",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 280.0, 184.0, 36.0, 20.0 ],
									"text" : "/ 100."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-72",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 88.0, 616.0, 52.0, 18.0 ],
									"text" : "ADSR+C"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-71",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 56.0, 608.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-70",
									"maxclass" : "newobj",
									"numinlets" : 6,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 56.0, 272.0, 94.0, 20.0 ],
									"text" : "pak 0. 0. 0. 0. 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-69",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 464.0, 144.0, 39.0, 18.0 ],
									"text" : "[-1. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-67",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 448.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-68",
									"index" : 8,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 448.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-59",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 408.0, 144.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-61",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 392.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-62",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 352.0, 144.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-63",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 336.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-64",
									"index" : 6,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 336.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-65",
									"index" : 7,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 392.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-54",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 296.0, 144.0, 19.0, 18.0 ],
									"text" : "%"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-55",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 280.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-56",
									"index" : 5,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 280.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-40",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 240.0, 144.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-41",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 224.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-44",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 184.0, 144.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-45",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 168.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-46",
									"index" : 3,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 168.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-49",
									"index" : 4,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 224.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-39",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 128.0, 144.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-36",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 112.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-35",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 72.0, 144.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-34",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 56.0, 160.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-31",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 56.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 11.0,
									"id" : "obj-107",
									"linecount" : 3,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 56.0, 24.0, 456.0, 43.0 ],
									"text" : "In this patcher we generate a function whose points are used to create an ADSR envelope. The envelope is drawn into a function object (for display) - therefore we're using the linear [0. 1.] output of live.dial. Time values are packed into a list, to be used by a curve~ object."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-19",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 296.0, 616.0, 53.0, 18.0 ],
									"text" : "Function"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-2",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 208.0, 456.0, 43.0, 18.0 ],
									"text" : "Display"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-1",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 160.0, 272.0, 72.0, 18.0 ],
									"text" : "< Sync data >"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-60",
									"index" : 2,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 264.0, 608.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-53",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 440.0, 88.0, 39.0, 18.0 ],
									"text" : "Curve"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-48",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 80.0, 88.0, 41.0, 18.0 ],
									"text" : "Attack"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-43",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 272.0, 88.0, 47.0, 18.0 ],
									"text" : "Sustain"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-38",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 192.0, 88.0, 40.0, 18.0 ],
									"text" : "Decay"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-37",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 352.0, 88.0, 48.0, 18.0 ],
									"text" : "Release"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-21",
									"index" : 2,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 112.0, 112.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-15",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "bang", "bang" ],
									"patching_rect" : [ 168.0, 456.0, 34.5, 20.0 ],
									"text" : "b"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-13",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "bang", "" ],
									"patching_rect" : [ 248.0, 296.0, 34.5, 20.0 ],
									"text" : "t b l"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-244",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 168.0, 528.0, 52.0, 20.0 ],
									"text" : "hidden 1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-243",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 168.0, 504.0, 64.0, 20.0 ],
									"text" : "mousefilter"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-242",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 184.0, 480.0, 52.0, 20.0 ],
									"text" : "hidden 0"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-194",
									"maxclass" : "newobj",
									"numinlets" : 5,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 248.0, 272.0, 83.0, 20.0 ],
									"text" : "pak 0. 0. 0. 0. 0."
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-194", 2 ],
									"order" : 0,
									"source" : [ "obj-103", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-70", 2 ],
									"order" : 1,
									"source" : [ "obj-103", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-204", 1 ],
									"source" : [ "obj-11", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-15", 0 ],
									"source" : [ "obj-13", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-196", 0 ],
									"source" : [ "obj-13", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-217", 0 ],
									"source" : [ "obj-14", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-71", 0 ],
									"source" : [ "obj-149", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-242", 0 ],
									"source" : [ "obj-15", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-243", 0 ],
									"source" : [ "obj-15", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-218", 0 ],
									"source" : [ "obj-17", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-13", 0 ],
									"source" : [ "obj-194", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-11", 0 ],
									"source" : [ "obj-196", 2 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-14", 0 ],
									"source" : [ "obj-196", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-17", 0 ],
									"source" : [ "obj-196", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-7", 0 ],
									"source" : [ "obj-196", 4 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-9", 0 ],
									"source" : [ "obj-196", 3 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-214", 0 ],
									"source" : [ "obj-197", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-23", 1 ],
									"source" : [ "obj-204", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-36", 0 ],
									"source" : [ "obj-21", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-23", 0 ],
									"source" : [ "obj-213", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-23", 2 ],
									"source" : [ "obj-214", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-218", 1 ],
									"order" : 1,
									"source" : [ "obj-217", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-233", 0 ],
									"order" : 0,
									"source" : [ "obj-217", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-237", 0 ],
									"source" : [ "obj-218", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-60", 0 ],
									"source" : [ "obj-22", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-214", 1 ],
									"source" : [ "obj-221", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-22", 0 ],
									"source" : [ "obj-23", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-204", 0 ],
									"source" : [ "obj-233", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-213", 0 ],
									"source" : [ "obj-237", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-60", 0 ],
									"source" : [ "obj-242", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-244", 0 ],
									"source" : [ "obj-243", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-60", 0 ],
									"source" : [ "obj-244", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-34", 0 ],
									"source" : [ "obj-31", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-70", 0 ],
									"source" : [ "obj-34", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-194", 3 ],
									"source" : [ "obj-36", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-194", 1 ],
									"source" : [ "obj-41", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-70", 1 ],
									"source" : [ "obj-45", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-45", 0 ],
									"source" : [ "obj-46", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-41", 0 ],
									"source" : [ "obj-49", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-103", 0 ],
									"source" : [ "obj-55", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-55", 0 ],
									"source" : [ "obj-56", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-194", 0 ],
									"source" : [ "obj-61", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-70", 3 ],
									"source" : [ "obj-63", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-63", 0 ],
									"source" : [ "obj-64", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-61", 0 ],
									"source" : [ "obj-65", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-194", 4 ],
									"order" : 1,
									"source" : [ "obj-67", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-70", 4 ],
									"order" : 2,
									"source" : [ "obj-67", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-73", 0 ],
									"order" : 0,
									"source" : [ "obj-67", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-67", 0 ],
									"source" : [ "obj-68", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-204", 2 ],
									"order" : 1,
									"source" : [ "obj-7", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-213", 1 ],
									"order" : 2,
									"source" : [ "obj-7", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-221", 0 ],
									"order" : 0,
									"source" : [ "obj-7", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-149", 0 ],
									"source" : [ "obj-70", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-70", 5 ],
									"source" : [ "obj-73", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-197", 0 ],
									"order" : 0,
									"source" : [ "obj-9", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-217", 1 ],
									"order" : 1,
									"source" : [ "obj-9", 0 ]
								}

							}
 ]
					}
,
					"patching_rect" : [ 685.483875870704651, 759.677424788475037, 243.0, 20.0 ],
					"saved_object_attributes" : 					{
						"description" : "",
						"digest" : "",
						"fontname" : "Arial Bold",
						"fontsize" : 10.0,
						"globalpatchername" : "",
						"tags" : ""
					}
,
					"text" : "p ADSR"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "ADSR curve.",
					"id" : "obj-36",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 909.677425861358643, 703.22581148147583, 51.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 340.0, 68.0, 48.0, 48.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Curve",
							"parameter_mapping_index" : 12,
							"parameter_mmax" : 1.0,
							"parameter_mmin" : -1.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Curve",
							"parameter_type" : 0,
							"parameter_unitstyle" : 1
						}

					}
,
					"varname" : "Curve"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Release time.",
					"id" : "obj-35",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 845.161296367645264, 703.22581148147583, 51.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 340.0, 119.0, 48.0, 48.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_exponent" : 4.333333,
							"parameter_initial" : [ 1.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Release",
							"parameter_mapping_index" : 16,
							"parameter_mmax" : 20000.0,
							"parameter_mmin" : 1.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Rel",
							"parameter_type" : 0,
							"parameter_unitstyle" : 2
						}

					}
,
					"varname" : "Release"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sustain level.",
					"id" : "obj-34",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 796.774199247360229, 703.22581148147583, 51.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 296.0, 119.0, 48.0, 48.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 100.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Sustain",
							"parameter_mapping_index" : 15,
							"parameter_mmax" : 100.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Sus",
							"parameter_type" : 0,
							"parameter_unitstyle" : 5
						}

					}
,
					"varname" : "Sustain"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Decay time..",
					"id" : "obj-32",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 750.00000536441803, 703.22581148147583, 51.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 252.0, 119.0, 48.0, 48.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_exponent" : 4.333333,
							"parameter_initial" : [ 1.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Decay",
							"parameter_mapping_index" : 14,
							"parameter_mmax" : 20000.0,
							"parameter_mmin" : 1.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Dec",
							"parameter_type" : 0,
							"parameter_unitstyle" : 2
						}

					}
,
					"varname" : "Decay"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Attack time.",
					"id" : "obj-30",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 685.483875870704651, 703.22581148147583, 51.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 208.0, 119.0, 48.0, 48.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_exponent" : 4.333333,
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Attack",
							"parameter_mapping_index" : 13,
							"parameter_mmax" : 20000.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Att",
							"parameter_type" : 0,
							"parameter_unitstyle" : 2
						}

					}
,
					"varname" : "Attack"
				}

			}
, 			{
				"box" : 				{
					"addpoints_with_curve" : [ 0.0, 0.0, 0, 0.0, 0.0, 1.0, 0, -0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 0, 0.0 ],
					"bgcolor" : [ 0.572549, 0.615686, 0.658824, 0.0 ],
					"classic_curve" : 1,
					"clickmove" : 0,
					"clicksustain" : 0,
					"domain" : 1.0,
					"gridcolor" : [ 0.5, 0.5, 0.5, 0.5 ],
					"hidden" : 1,
					"id" : "obj-183",
					"legend" : 0,
					"linecolor" : [ 1.0, 1.0, 1.0, 1.0 ],
					"maxclass" : "function",
					"mode" : 1,
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "float", "", "", "bang" ],
					"outputmode" : 1,
					"parameter_enable" : 0,
					"patching_rect" : [ 909.677425861358643, 790.322586297988892, 100.0, 50.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 12.0, 30.0, 313.0, 74.0 ],
					"textcolor" : [ 0.0, 0.019608, 0.078431, 0.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-126",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 612.9032301902771, 759.677424788475037, 49.0, 20.0 ],
					"text" : "Loop $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-124",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 477.419358253479004, 759.677424788475037, 43.0, 20.0 ],
					"text" : "End $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-122",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 412.903228759765625, 759.677424788475037, 47.0, 20.0 ],
					"text" : "Start $1"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-80",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 487.096777677536011, 719.354843854904175, 51.0, 18.0 ],
					"text" : "End (ms)"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial",
					"fontsize" : 10.0,
					"id" : "obj-84",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 424.193551421165466, 719.354843854904175, 54.0, 18.0 ],
					"text" : "Start (ms)"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"format" : 6,
					"id" : "obj-89",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 477.419358253479004, 735.48387622833252, 56.0, 20.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"format" : 6,
					"id" : "obj-93",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 412.903228759765625, 735.48387622833252, 56.0, 20.0 ]
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Snap selection to zeros.",
					"appearance" : 1,
					"automation" : "off",
					"automationon" : "on",
					"id" : "obj-16",
					"maxclass" : "live.text",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 445.161293506622314, 575.806455731391907, 48.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 160.0, 136.0, 45.0, 14.0 ],
					"rounded" : 64.0,
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Snap",
							"parameter_mapping_index" : 18,
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Snap",
							"parameter_type" : 2
						}

					}
,
					"text" : "Snap",
					"texton" : "Snap",
					"varname" : "Snap"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-176",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 6,
							"revision" : 5,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 343.0, 433.0, 483.0, 510.0 ],
						"bglocked" : 0,
						"openinpresentation" : 0,
						"default_fontsize" : 10.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial Bold",
						"gridonopen" : 1,
						"gridsize" : [ 8.0, 8.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"boxes" : [ 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-48",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 272.0, 312.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-24",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 200.0, 312.0, 36.0, 18.0 ],
									"text" : "[0. 1.]"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-61",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 347.0, 224.0, 64.0, 18.0 ],
									"text" : "< Sync data"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-58",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 112.0, 248.0, 43.0, 18.0 ],
									"text" : "Update"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-57",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 107.0, 68.0, 35.0, 18.0 ],
									"text" : "Snap"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-55",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 132.0, 128.0, 38.0, 18.0 ],
									"text" : "On/off"
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-56",
									"maxclass" : "toggle",
									"numinlets" : 1,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 112.0, 128.0, 20.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-40",
									"linecount" : 2,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 24.0, 276.0, 75.0, 29.0 ],
									"text" : "To waveform~ objects"
								}

							}
, 							{
								"box" : 								{
									"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-52",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 24.0, 256.0, 77.0, 20.0 ],
									"text" : "s ---Waveform"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-43",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 2,
									"outlettype" : [ "bang", "" ],
									"patching_rect" : [ 112.0, 224.0, 33.0, 20.0 ],
									"text" : "sel 1"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-98",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "int", "int" ],
									"patching_rect" : [ 112.0, 176.0, 34.5, 20.0 ],
									"text" : "t i i"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-44",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "int" ],
									"patching_rect" : [ 112.0, 152.0, 32.5, 20.0 ],
									"text" : "* 2"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-47",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 24.0, 224.0, 57.0, 20.0 ],
									"text" : "snapto $1"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-39",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 112.0, 88.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-25",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 254.0, 456.0, 30.0, 18.0 ],
									"text" : "End"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-27",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 180.0, 456.0, 34.0, 18.0 ],
									"text" : "Start"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-21",
									"index" : 2,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 256.0, 424.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-20",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 184.0, 424.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 11.0,
									"id" : "obj-1",
									"linecount" : 2,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 176.0, 24.0, 197.0, 31.0 ],
									"text" : "Calculate some position values relatively to the duration of the sample."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-10",
									"linecount" : 2,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 336.0, 144.0, 96.0, 29.0 ],
									"text" : "From LoadSample subpatcher"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-8",
									"linecount" : 3,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 136.0, 352.0, 44.0, 40.0 ],
									"text" : "Map to sample length >"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-160",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 254.0, 68.0, 30.0, 18.0 ],
									"text" : "End"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-162",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 180.0, 68.0, 34.0, 18.0 ],
									"text" : "Start"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-164",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 304.0, 128.0, 19.0, 18.0 ],
									"text" : "%"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-165",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 232.0, 128.0, 19.0, 18.0 ],
									"text" : "%"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-147",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 256.0, 200.0, 36.0, 20.0 ],
									"text" : "/ 100."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-143",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 184.0, 200.0, 36.0, 20.0 ],
									"text" : "/ 100."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-139",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 256.0, 128.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-138",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 184.0, 128.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-67",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 256.0, 176.0, 61.0, 20.0 ],
									"text" : "clip 0. 100."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-54",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 184.0, 176.0, 61.0, 20.0 ],
									"text" : "clip 0. 100."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-16",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 272.0, 280.0, 28.0, 18.0 ],
									"text" : "End"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-15",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 200.0, 280.0, 32.0, 18.0 ],
									"text" : "Start"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-100",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 288.0, 360.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-101",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 256.0, 376.0, 56.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-103",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 256.0, 352.0, 32.5, 20.0 ],
									"text" : "* 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-45",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 256.0, 296.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-42",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 184.0, 296.0, 48.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-38",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 344.0, 280.0, 41.0, 18.0 ],
									"text" : "Length"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-36",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 216.0, 360.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-37",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 184.0, 376.0, 56.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-34",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 384.0, 296.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-35",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 328.0, 296.0, 56.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-32",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 184.0, 352.0, 32.5, 20.0 ],
									"text" : "* 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-30",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 3,
									"outlettype" : [ "float", "float", "float" ],
									"patching_rect" : [ 184.0, 256.0, 163.0, 20.0 ],
									"text" : "unpack 0. 0. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-29",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 184.0, 224.0, 163.0, 20.0 ],
									"text" : "pak 0. 1. 0."
								}

							}
, 							{
								"box" : 								{
									"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-41",
									"maxclass" : "newobj",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 328.0, 176.0, 96.0, 20.0 ],
									"text" : "r ---SampleLength"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-26",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 384.0, 200.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-28",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 328.0, 199.0, 56.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-169",
									"index" : 2,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 184.0, 88.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-170",
									"index" : 3,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 256.0, 88.0, 25.0, 25.0 ]
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-21", 0 ],
									"source" : [ "obj-101", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-101", 0 ],
									"source" : [ "obj-103", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-54", 0 ],
									"order" : 1,
									"source" : [ "obj-138", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-67", 1 ],
									"order" : 0,
									"source" : [ "obj-138", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-54", 2 ],
									"order" : 1,
									"source" : [ "obj-139", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-67", 0 ],
									"order" : 0,
									"source" : [ "obj-139", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-29", 0 ],
									"source" : [ "obj-143", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-29", 1 ],
									"source" : [ "obj-147", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-138", 0 ],
									"source" : [ "obj-169", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-139", 0 ],
									"source" : [ "obj-170", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-29", 2 ],
									"source" : [ "obj-28", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-30", 0 ],
									"source" : [ "obj-29", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-35", 0 ],
									"source" : [ "obj-30", 2 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-42", 0 ],
									"source" : [ "obj-30", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-45", 0 ],
									"source" : [ "obj-30", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-37", 0 ],
									"source" : [ "obj-32", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-103", 1 ],
									"order" : 0,
									"source" : [ "obj-35", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-32", 1 ],
									"order" : 1,
									"source" : [ "obj-35", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-20", 0 ],
									"source" : [ "obj-37", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-56", 0 ],
									"source" : [ "obj-39", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-28", 0 ],
									"source" : [ "obj-41", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-32", 0 ],
									"source" : [ "obj-42", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-29", 0 ],
									"source" : [ "obj-43", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-98", 0 ],
									"source" : [ "obj-44", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-103", 0 ],
									"source" : [ "obj-45", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-52", 0 ],
									"source" : [ "obj-47", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-143", 0 ],
									"source" : [ "obj-54", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-44", 0 ],
									"source" : [ "obj-56", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-147", 0 ],
									"source" : [ "obj-67", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-43", 0 ],
									"source" : [ "obj-98", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-47", 0 ],
									"source" : [ "obj-98", 1 ]
								}

							}
 ],
						"styles" : [ 							{
								"name" : "AudioStatus_Menu",
								"default" : 								{
									"bgfillcolor" : 									{
										"angle" : 270.0,
										"autogradient" : 0,
										"color" : [ 0.294118, 0.313726, 0.337255, 1 ],
										"color1" : [ 0.454902, 0.462745, 0.482353, 0.0 ],
										"color2" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
										"proportion" : 0.39,
										"type" : "color"
									}

								}
,
								"parentstyle" : "",
								"multi" : 0
							}
 ]
					}
,
					"patching_rect" : [ 445.161293506622314, 598.387101054191589, 115.0, 20.0 ],
					"saved_object_attributes" : 					{
						"description" : "",
						"digest" : "",
						"fontname" : "Arial Bold",
						"fontsize" : 10.0,
						"globalpatchername" : "",
						"tags" : ""
					}
,
					"text" : "p Selection"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Enables/disables loop.",
					"appearance" : 1,
					"automation" : "off",
					"automationon" : "on",
					"id" : "obj-11",
					"maxclass" : "live.text",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 612.9032301902771, 735.48387622833252, 48.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 160.0, 119.0, 45.0, 14.0 ],
					"rounded" : 64.0,
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_enum" : [ "off", "on" ],
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "EnableLoop",
							"parameter_mapping_index" : 17,
							"parameter_mmax" : 100.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Enable",
							"parameter_type" : 0,
							"parameter_unitstyle" : 5
						}

					}
,
					"text" : "Loop",
					"texton" : "Loop",
					"varname" : "EnableLoop"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "End position relatively to the sample's length.",
					"id" : "obj-10",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 541.935487747192383, 543.548390984535217, 48.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 108.0, 119.0, 48.0, 48.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 100 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "End",
							"parameter_mapping_index" : 7,
							"parameter_mmax" : 100.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "End",
							"parameter_type" : 0,
							"parameter_unitstyle" : 5
						}

					}
,
					"varname" : "End"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Start position relatively to the sample's length.",
					"id" : "obj-4",
					"maxclass" : "live.dial",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 493.548390626907349, 543.548390984535217, 48.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 62.0, 119.0, 48.0, 48.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Start",
							"parameter_mapping_index" : 6,
							"parameter_mmax" : 100.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Start",
							"parameter_type" : 0,
							"parameter_unitstyle" : 5
						}

					}
,
					"varname" : "Start"
				}

			}
, 			{
				"box" : 				{
					"angle" : 0.0,
					"bgcolor" : [ 0.0, 0.0, 0.0, 1.0 ],
					"id" : "obj-161",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 741.935489177703857, 1190.322589159011841, 16.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 16.0, 108.0, 300.0, 8.0 ],
					"proportion" : 0.39,
					"prototypename" : "M4L.horizontal-black",
					"rounded" : 0
				}

			}
, 			{
				"box" : 				{
					"angle" : 0.0,
					"bgcolor" : [ 0.094118, 0.113725, 0.137255, 0.0 ],
					"border" : 3,
					"bordercolor" : [ 0.0, 0.0, 0.0, 1.0 ],
					"id" : "obj-6",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 766.129037737846375, 1190.322589159011841, 16.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 15.0, 21.0, 306.0, 80.0 ],
					"proportion" : 0.39
				}

			}
, 			{
				"box" : 				{
					"allowdrag" : 0,
					"annotation" : "Selects audio.",
					"beats" : 0,
					"bgcolor" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
					"buffername" : "---samplerbuffer",
					"grid" : 0.125,
					"gridcolor" : [ 0.65098, 0.666667, 0.662745, 1.0 ],
					"id" : "obj-7",
					"maxclass" : "waveform~",
					"numinlets" : 5,
					"numoutlets" : 6,
					"outlettype" : [ "float", "float", "float", "float", "list", "" ],
					"patching_rect" : [ 285.483873009681702, 638.709681987762451, 339.0, 73.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 16.0, 23.0, 304.0, 76.0 ],
					"selectioncolor" : [ 1.0, 0.788235, 0.027451, 0.5 ],
					"vticks" : 0,
					"waveformcolor" : [ 0.65098, 0.666667, 0.662745, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Drag to zoom vertical audio's view in and out.",
					"bgcolor" : [ 0.552941, 0.552941, 0.552941, 1.0 ],
					"contdata" : 1,
					"id" : "obj-102",
					"maxclass" : "multislider",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 212.90322732925415, 638.709681987762451, 24.0, 48.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 324.0, 24.0, 8.0, 72.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0 ],
							"parameter_initial_enable" : 1,
							"parameter_invisible" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "ZoomV",
							"parameter_mmax" : 1.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Zoom",
							"parameter_type" : 0,
							"parameter_unitstyle" : 1
						}

					}
,
					"setminmax" : [ 0.0, 1.0 ],
					"setstyle" : 1,
					"slidercolor" : [ 0.0, 1.0, 1.0, 0.5 ],
					"varname" : "ZoomV"
				}

			}
, 			{
				"box" : 				{
					"activebgcolor" : [ 0.6, 0.6, 0.6, 0.0 ],
					"activetextcolor" : [ 0.74902, 0.74902, 0.74902, 1.0 ],
					"annotation" : "Resets zooming.",
					"automation" : "arm",
					"automationon" : "trig",
					"bgcolor" : [ 0.6, 0.6, 0.6, 0.0 ],
					"bordercolor" : [ 0.552941, 0.552941, 0.552941, 1.0 ],
					"fontsize" : 8.0,
					"id" : "obj-82",
					"maxclass" : "live.text",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 212.90322732925415, 575.806455731391907, 24.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 319.0, 99.0, 12.0, 12.0 ],
					"prototypename" : "numbers.default",
					"saved_attribute_attributes" : 					{
						"activebgcolor" : 						{
							"expression" : ""
						}
,
						"activetextcolor" : 						{
							"expression" : ""
						}
,
						"bgcolor" : 						{
							"expression" : ""
						}
,
						"bordercolor" : 						{
							"expression" : ""
						}
,
						"textcolor" : 						{
							"expression" : ""
						}
,
						"valueof" : 						{
							"parameter_enum" : [ "arm", "trig" ],
							"parameter_invisible" : 2,
							"parameter_linknames" : 1,
							"parameter_longname" : "ZoomInit",
							"parameter_mmax" : 1,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Init",
							"parameter_type" : 2
						}

					}
,
					"text" : "Z",
					"textcolor" : [ 0.74902, 0.74902, 0.74902, 1.0 ],
					"texton" : "Z",
					"varname" : "ZoomInit"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Drag to zoom horizontal audio's view in and out. Shift-clicking extends the range instead of replacing it. Command-clicking (Macintosh) or Control-double-clicking (Windows) and dragging shifts the current range values up or down. Option-clicking (Macintosh) or Alt-clicking (Windows) and dragging up or down expands or shrinks the currently selected range. Command-double-clicking (Macintosh) or Control-double-clicking (Windows) selects the entire range.",
					"drawline" : 0,
					"fgcolor" : [ 0.0, 1.0, 1.0, 0.8 ],
					"floatoutput" : 1,
					"id" : "obj-39",
					"listmode" : 1,
					"maxclass" : "rslider",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 285.483873009681702, 551.61290717124939, 104.0, 10.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 18.0, 105.0, 298.0, 11.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0.0, 1.0 ],
							"parameter_initial_enable" : 1,
							"parameter_invisible" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "ZoomH",
							"parameter_modmode" : 0,
							"parameter_shortname" : "ZoomH",
							"parameter_type" : 3
						}

					}
,
					"size" : 1.0,
					"varname" : "ZoomH"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-5",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 212.90322732925415, 614.516133427619934, 24.0, 20.0 ],
					"text" : "0."
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-135",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 6,
							"revision" : 5,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 134.0, 172.0, 265.0, 319.0 ],
						"bglocked" : 0,
						"openinpresentation" : 0,
						"default_fontsize" : 10.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial Bold",
						"gridonopen" : 1,
						"gridsize" : [ 8.0, 8.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"boxes" : [ 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-4",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 104.0, 280.0, 84.0, 18.0 ],
									"text" : "Loop waveform"
								}

							}
, 							{
								"box" : 								{
									"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-58",
									"maxclass" : "newobj",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 128.0, 192.0, 75.0, 20.0 ],
									"text" : "r ---Waveform"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-1",
									"index" : 2,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 128.0, 248.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-50",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 144.0, 168.0, 94.0, 18.0 ],
									"text" : "Remote messages"
								}

							}
, 							{
								"box" : 								{
									"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-52",
									"maxclass" : "newobj",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 128.0, 144.0, 100.0, 20.0 ],
									"text" : "r ---LoopWaveform"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-3",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 40.0, 280.0, 58.0, 18.0 ],
									"text" : "Waveform"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-2",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 88.0, 24.0, 38.0, 18.0 ],
									"text" : "Slider"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-81",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 56.0, 72.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-71",
									"maxclass" : "newobj",
									"numinlets" : 6,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 56.0, 96.0, 91.0, 20.0 ],
									"text" : "scale 0. 1. 1. 0.01"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-65",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 56.0, 120.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-57",
									"maxclass" : "message",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 56.0, 144.0, 56.0, 20.0 ],
									"text" : "vzoom $1"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-133",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 56.0, 24.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-134",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 56.0, 248.0, 25.0, 25.0 ]
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-81", 0 ],
									"source" : [ "obj-133", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-134", 0 ],
									"source" : [ "obj-52", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-134", 0 ],
									"source" : [ "obj-57", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-1", 0 ],
									"order" : 0,
									"source" : [ "obj-58", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-134", 0 ],
									"order" : 1,
									"source" : [ "obj-58", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-57", 0 ],
									"source" : [ "obj-65", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-65", 0 ],
									"source" : [ "obj-71", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-71", 0 ],
									"source" : [ "obj-81", 0 ]
								}

							}
 ]
					}
,
					"patching_rect" : [ 212.90322732925415, 695.161295294761658, 53.0, 20.0 ],
					"saved_object_attributes" : 					{
						"description" : "",
						"digest" : "",
						"fontname" : "Arial Bold",
						"fontsize" : 10.0,
						"globalpatchername" : "",
						"tags" : ""
					}
,
					"text" : "p ZoomV"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-132",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "int", "int" ],
					"patcher" : 					{
						"fileversion" : 1,
						"appversion" : 						{
							"major" : 8,
							"minor" : 6,
							"revision" : 5,
							"architecture" : "x64",
							"modernui" : 1
						}
,
						"classnamespace" : "box",
						"rect" : [ 836.0, 85.0, 268.0, 492.0 ],
						"bglocked" : 0,
						"openinpresentation" : 0,
						"default_fontsize" : 10.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial Bold",
						"gridonopen" : 1,
						"gridsize" : [ 8.0, 8.0 ],
						"gridsnaponopen" : 1,
						"objectsnaponopen" : 1,
						"statusbarvisible" : 2,
						"toolbarvisible" : 1,
						"lefttoolbarpinned" : 0,
						"toptoolbarpinned" : 0,
						"righttoolbarpinned" : 0,
						"bottomtoolbarpinned" : 0,
						"toolbars_unpinned_last_save" : 0,
						"tallnewobj" : 0,
						"boxanimatetime" : 200,
						"enablehscroll" : 1,
						"enablevscroll" : 1,
						"devicewidth" : 0.0,
						"description" : "",
						"digest" : "",
						"tags" : "",
						"style" : "",
						"subpatcher_template" : "",
						"assistshowspatchername" : 0,
						"boxes" : [ 							{
								"box" : 								{
									"id" : "obj-8",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 2,
									"outlettype" : [ "int", "int" ],
									"patching_rect" : [ 63.5, 344.0, 73.0, 20.0 ],
									"text" : "fswap"
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-7",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 117.5, 282.0, 29.5, 20.0 ],
									"text" : "!- 0."
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-6",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 128.0, 238.0, 29.5, 20.0 ],
									"text" : "* 0."
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-5",
									"maxclass" : "newobj",
									"numinlets" : 1,
									"numoutlets" : 3,
									"outlettype" : [ "float", "float", "float" ],
									"patching_rect" : [ 64.0, 192.0, 93.5, 20.0 ],
									"text" : "unpack 0. 0. 0."
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-4",
									"maxclass" : "newobj",
									"numinlets" : 2,
									"numoutlets" : 1,
									"outlettype" : [ "float" ],
									"patching_rect" : [ 64.0, 238.0, 29.5, 20.0 ],
									"text" : "* 0."
								}

							}
, 							{
								"box" : 								{
									"id" : "obj-2",
									"maxclass" : "newobj",
									"numinlets" : 3,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 64.0, 163.0, 61.0, 20.0 ],
									"text" : "pak 0. 1. 0."
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 11.0,
									"id" : "obj-57",
									"linecount" : 4,
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 28.0, 16.0, 215.0, 56.0 ],
									"text" : "We're using rslider to perform zooming. Start and Length values are scaled to the current sample's length, and control the display window of the main waveform~"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-9",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 111.0, 440.0, 44.0, 18.0 ],
									"text" : "Length"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-10",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 59.5, 440.0, 33.0, 18.0 ],
									"text" : "Start"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-3",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 54.0, 83.0, 45.0, 18.0 ],
									"text" : "ZoomH"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-1",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 173.5, 307.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-94",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 117.5, 307.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-93",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 64.0, 307.0, 50.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial",
									"fontsize" : 10.0,
									"id" : "obj-88",
									"maxclass" : "comment",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 162.0, 129.0, 24.0, 18.0 ],
									"text" : "ms"
								}

							}
, 							{
								"box" : 								{
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"format" : 6,
									"id" : "obj-89",
									"maxclass" : "flonum",
									"numinlets" : 1,
									"numoutlets" : 2,
									"outlettype" : [ "", "bang" ],
									"parameter_enable" : 0,
									"patching_rect" : [ 106.0, 129.0, 56.0, 20.0 ]
								}

							}
, 							{
								"box" : 								{
									"color" : [ 0.545098, 0.85098, 0.592157, 1.0 ],
									"fontname" : "Arial Bold",
									"fontsize" : 10.0,
									"id" : "obj-86",
									"maxclass" : "newobj",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 106.0, 105.0, 96.0, 20.0 ],
									"text" : "r ---SampleLength"
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-124",
									"index" : 1,
									"maxclass" : "inlet",
									"numinlets" : 0,
									"numoutlets" : 1,
									"outlettype" : [ "" ],
									"patching_rect" : [ 64.0, 104.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-130",
									"index" : 1,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 63.5, 408.0, 25.0, 25.0 ]
								}

							}
, 							{
								"box" : 								{
									"comment" : "",
									"id" : "obj-131",
									"index" : 2,
									"maxclass" : "outlet",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 117.5, 408.0, 25.0, 25.0 ]
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"destination" : [ "obj-2", 0 ],
									"source" : [ "obj-124", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-5", 0 ],
									"source" : [ "obj-2", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-7", 0 ],
									"order" : 0,
									"source" : [ "obj-4", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-93", 0 ],
									"order" : 1,
									"source" : [ "obj-4", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-4", 1 ],
									"order" : 1,
									"source" : [ "obj-5", 2 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-4", 0 ],
									"source" : [ "obj-5", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-6", 1 ],
									"order" : 0,
									"source" : [ "obj-5", 2 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-6", 0 ],
									"source" : [ "obj-5", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-7", 1 ],
									"source" : [ "obj-6", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-94", 0 ],
									"source" : [ "obj-7", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-130", 0 ],
									"source" : [ "obj-8", 1 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-131", 0 ],
									"source" : [ "obj-8", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-89", 0 ],
									"source" : [ "obj-86", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-2", 2 ],
									"source" : [ "obj-89", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-8", 0 ],
									"source" : [ "obj-93", 0 ]
								}

							}
, 							{
								"patchline" : 								{
									"destination" : [ "obj-8", 1 ],
									"source" : [ "obj-94", 0 ]
								}

							}
 ],
						"styles" : [ 							{
								"name" : "AudioStatus_Menu",
								"default" : 								{
									"bgfillcolor" : 									{
										"angle" : 270.0,
										"autogradient" : 0,
										"color" : [ 0.294118, 0.313726, 0.337255, 1 ],
										"color1" : [ 0.454902, 0.462745, 0.482353, 0.0 ],
										"color2" : [ 0.290196, 0.309804, 0.301961, 1.0 ],
										"proportion" : 0.39,
										"type" : "color"
									}

								}
,
								"parentstyle" : "",
								"multi" : 0
							}
 ]
					}
,
					"patching_rect" : [ 285.483873009681702, 575.806455731391907, 54.0, 20.0 ],
					"saved_object_attributes" : 					{
						"description" : "",
						"digest" : "",
						"fontname" : "Arial Bold",
						"fontsize" : 10.0,
						"globalpatchername" : "",
						"tags" : ""
					}
,
					"text" : "p ZoomH"
				}

			}
, 			{
				"box" : 				{
					"annotation" : "Sets the amount of loop fade.",
					"id" : "obj-12",
					"maxclass" : "live.numbox",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "float" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 550.000003933906555, 735.48387622833252, 48.0, 15.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 14.0, 151.0, 40.0, 15.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0.0 ],
							"parameter_initial_enable" : 1,
							"parameter_linknames" : 1,
							"parameter_longname" : "Fade",
							"parameter_mapping_index" : 9,
							"parameter_mmax" : 100.0,
							"parameter_modmode" : 0,
							"parameter_shortname" : "Fade",
							"parameter_type" : 0,
							"parameter_unitstyle" : 5
						}

					}
,
					"varname" : "Fade"
				}

			}
, 			{
				"box" : 				{
					"fontname" : "Arial Bold",
					"fontsize" : 10.0,
					"id" : "obj-20",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 341.935486316680908, 879.032264351844788, 60.0, 20.0 ],
					"text" : "allnotesoff"
				}

			}
, 			{
				"box" : 				{
					"angle" : 0.0,
					"background" : 1,
					"bgcolor" : [ 0.0, 0.0, 0.0, 1.0 ],
					"id" : "obj-159",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 717.74194061756134, 1190.322589159011841, 16.0, 16.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 8.0, 8.0, 327.0, 108.0 ],
					"proportion" : 0.39,
					"prototypename" : "M4L.horizontal-black",
					"rounded" : 16
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-176", 2 ],
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 1 ],
					"order" : 1,
					"source" : [ "obj-101", 6 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-106", 0 ],
					"order" : 0,
					"source" : [ "obj-101", 6 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-61", 0 ],
					"source" : [ "obj-101", 8 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-135", 0 ],
					"source" : [ "obj-102", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-92", 0 ],
					"source" : [ "obj-103", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-101", 0 ],
					"source" : [ "obj-105", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-101", 0 ],
					"source" : [ "obj-105", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-2", 0 ],
					"source" : [ "obj-107", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-109", 0 ],
					"source" : [ "obj-108", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-165", 0 ],
					"source" : [ "obj-109", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-126", 0 ],
					"source" : [ "obj-11", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-112", 0 ],
					"source" : [ "obj-110", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-108", 0 ],
					"source" : [ "obj-111", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-111", 0 ],
					"source" : [ "obj-112", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 0 ],
					"source" : [ "obj-114", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 0 ],
					"source" : [ "obj-116", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-62", 1 ],
					"source" : [ "obj-117", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-62", 0 ],
					"source" : [ "obj-117", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 0 ],
					"source" : [ "obj-12", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-112", 1 ],
					"order" : 1,
					"source" : [ "obj-121", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-123", 0 ],
					"order" : 0,
					"source" : [ "obj-121", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 1 ],
					"source" : [ "obj-122", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-117", 2 ],
					"source" : [ "obj-123", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 1 ],
					"source" : [ "obj-124", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-110", 0 ],
					"source" : [ "obj-125", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 1 ],
					"source" : [ "obj-126", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 0 ],
					"source" : [ "obj-127", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-92", 0 ],
					"source" : [ "obj-128", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 1 ],
					"source" : [ "obj-132", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-132", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-135", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-176", 0 ],
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-117", 0 ],
					"source" : [ "obj-165", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 1 ],
					"source" : [ "obj-17", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 3 ],
					"source" : [ "obj-176", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 2 ],
					"source" : [ "obj-176", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-20", 0 ],
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 10 ],
					"source" : [ "obj-192", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-198", 0 ],
					"source" : [ "obj-196", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 1 ],
					"source" : [ "obj-198", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-100", 0 ],
					"order" : 0,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-3", 0 ],
					"order" : 1,
					"source" : [ "obj-2", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 0 ],
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 1 ],
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 0 ],
					"source" : [ "obj-24", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-21", 0 ],
					"source" : [ "obj-25", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.0, 0.0, 0.0, 0.0 ],
					"destination" : [ "obj-38", 0 ],
					"source" : [ "obj-25", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.0, 0.0, 0.0, 0.0 ],
					"destination" : [ "obj-42", 0 ],
					"order" : 0,
					"source" : [ "obj-25", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.0, 0.0, 0.0, 0.0 ],
					"destination" : [ "obj-43", 0 ],
					"source" : [ "obj-25", 4 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.0, 0.0, 0.0, 0.0 ],
					"destination" : [ "obj-43", 0 ],
					"order" : 1,
					"source" : [ "obj-25", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.0, 0.0, 0.0, 0.0 ],
					"destination" : [ "obj-52", 0 ],
					"source" : [ "obj-25", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.0, 0.0, 0.0, 0.0 ],
					"destination" : [ "obj-53", 0 ],
					"order" : 2,
					"source" : [ "obj-25", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"color" : [ 0.0, 0.0, 0.0, 0.0 ],
					"destination" : [ "obj-57", 0 ],
					"order" : 3,
					"source" : [ "obj-25", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-90", 0 ],
					"source" : [ "obj-26", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-44", 0 ],
					"source" : [ "obj-28", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-3", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 1 ],
					"source" : [ "obj-30", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 0 ],
					"source" : [ "obj-30", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 3 ],
					"source" : [ "obj-32", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 2 ],
					"source" : [ "obj-32", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-90", 0 ],
					"source" : [ "obj-33", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 4 ],
					"source" : [ "obj-34", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 6 ],
					"source" : [ "obj-35", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 5 ],
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-63", 7 ],
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 2 ],
					"source" : [ "obj-38", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-132", 0 ],
					"source" : [ "obj-39", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-176", 1 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 1 ],
					"order" : 0,
					"source" : [ "obj-40", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 0 ],
					"order" : 1,
					"source" : [ "obj-40", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 0 ],
					"source" : [ "obj-40", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 8 ],
					"source" : [ "obj-42", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 7 ],
					"source" : [ "obj-43", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-38", 0 ],
					"order" : 2,
					"source" : [ "obj-44", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-50", 0 ],
					"order" : 1,
					"source" : [ "obj-44", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-52", 0 ],
					"order" : 0,
					"source" : [ "obj-44", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"source" : [ "obj-45", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-118", 0 ],
					"source" : [ "obj-45", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 3 ],
					"source" : [ "obj-49", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-102", 0 ],
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 4 ],
					"source" : [ "obj-50", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 6 ],
					"source" : [ "obj-52", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 5 ],
					"source" : [ "obj-53", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-40", 1 ],
					"source" : [ "obj-55", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-40", 0 ],
					"source" : [ "obj-55", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 1 ],
					"source" : [ "obj-57", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-72", 0 ],
					"source" : [ "obj-59", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-114", 0 ],
					"source" : [ "obj-62", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-183", 0 ],
					"source" : [ "obj-63", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-65", 0 ],
					"source" : [ "obj-63", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-55", 1 ],
					"source" : [ "obj-65", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"order" : 6,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-126", 0 ],
					"order" : 5,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-196", 0 ],
					"order" : 9,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 0 ],
					"order" : 4,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-32", 0 ],
					"order" : 3,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-34", 0 ],
					"order" : 2,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-35", 0 ],
					"order" : 1,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-36", 0 ],
					"order" : 0,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-89", 0 ],
					"order" : 7,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-93", 0 ],
					"order" : 8,
					"source" : [ "obj-66", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-89", 0 ],
					"source" : [ "obj-7", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-93", 0 ],
					"source" : [ "obj-7", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-105", 0 ],
					"order" : 1,
					"source" : [ "obj-72", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-68", 0 ],
					"order" : 0,
					"source" : [ "obj-72", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 9 ],
					"source" : [ "obj-73", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-90", 0 ],
					"source" : [ "obj-75", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"order" : 1,
					"source" : [ "obj-82", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-9", 0 ],
					"order" : 0,
					"source" : [ "obj-82", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-90", 0 ],
					"source" : [ "obj-87", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-90", 0 ],
					"source" : [ "obj-88", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-124", 0 ],
					"source" : [ "obj-89", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-39", 0 ],
					"source" : [ "obj-9", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-56", 0 ],
					"source" : [ "obj-90", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-72", 0 ],
					"order" : 0,
					"source" : [ "obj-90", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-81", 0 ],
					"order" : 1,
					"source" : [ "obj-90", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-90", 0 ],
					"source" : [ "obj-92", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-122", 0 ],
					"source" : [ "obj-93", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"source" : [ "obj-96", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-92", 0 ],
					"source" : [ "obj-97", 0 ]
				}

			}
 ]
	}

}
