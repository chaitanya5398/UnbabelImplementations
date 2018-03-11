This is a file about the various files in this folder.

The Folder has three kinds of files.

1) .txt files
   	a) OK/BAD outputs for each words of sentense.
	b) Hter Scores of each sentense.

2) nueralQE*.py - The various files which load the data and run model and generate .txt files.

3) data_load*.py - These files take the data and present in manners suitable for various models.

The Functions used in the .py files.

FileName: nueralQE*.py
Functions

	1) The custom loss functions are located at the top.
	2) create_model(): Has the model definition and compilation.
	3) batch_wise_operate(): This trains the model in batches made according to sentnese lengths.
	4) The main function has the prediction part of code and saving them.

FileName: data_load*.py
Functions

	1) make_align_dict(): Gets the alignments dictionary, Key = TargetWordIndex, Value = List Of Source Word Indices that are aligned to it.
	2) get_target_embedding(): It gives the polyglot embedding of the taget sentense words.
	3) get_source_embedding(): It gives the polyglot embeddings of the words aligned to the target word.
	4) get_pos_embedding(): gives the pos embeddings.
	5) get_sentense_inputs(): It gives the list of embeddings/ list of lists of embeddings for sentense.
	6) get_data_mats(): returns the data of all the sentenses.

	
