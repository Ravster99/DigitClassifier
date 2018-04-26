import java.util.*;
/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */
public class NNImpl {
    private ArrayList<Node> inputNodes; 	//list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    //list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set
    private int hiddenNodeCounter; //Total number of hidden nodes
    private Double[][] outputWeights;

    /**
     * This constructor creates the nodes necessary for the neural network 
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;
        this.hiddenNodeCounter = hiddenNodeCount+1; //Including the bias.
        this.outputWeights = outputWeights;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++){
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }
	/**
     * Get the prediction from the neural network for a single instance
     * Return the index with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */
    public int predict(Instance instance) {
		for (int j = 0; j < instance.attributes.size(); j++) {//Assigning value of each attribute to the node.
			inputNodes.get(j).setInput(instance.attributes.get(j));
		}
		double denominator = 0;
		double output[] = new double[instance.classValues.size()];
		for (int k = 0; k < instance.classValues.size(); k++) {
			output[k] =  Math.exp(outputNodes.get(k).getOutput()); 
			denominator =  denominator + output[k];
		}
		//Determining the output values.
		for (int k = 0; k < instance.classValues.size(); k++) {
			output[k] = output[k]/denominator;
		}
		//Determining the index in the array for prediction.
		int return_integer = 0;
		double max = 0.0;
		for (int k = 0; k < instance.classValues.size(); k++) {
			if (output[k] > max) {
				return_integer = k;
				max = output[k];
			}
		}
		//Returning the index.
        return return_integer;
    }
    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */    
    public void train() {
    	for (int e = 0; e < maxEpoch; e++) { //Running the loop for each of the epoch.
    		Collections.shuffle(trainingSet, random);
    		double sum_epoch = 0.0;
	     	for (int i = 0; i < trainingSet.size(); i++) { //Training for each of the instance in the training example.
	    		for (int j = 0; j < trainingSet.get(0).attributes.size(); j++) {//Assigning value of each attribute to the node.
	    			inputNodes.get(j).setInput(trainingSet.get(i).attributes.get(j));
	    		}
	    		//Creating a immutable copy of the weight for updating the hidden layer.
	    		Double[][] outputWeights_copy = outputWeights.clone();
	            for (int m = 0; m < outputWeights.length; m++) {
	                for (int n = 0; n < outputWeights[m].length; n++) {
	                    outputWeights_copy[m][n] = outputNodes.get(m).parents.get(n).weight;
	                }
	            }
	    		double denominator, delta_j;
	    		denominator = delta_j = 0;
	    		double output[] = new double[trainingSet.get(0).classValues.size()];
	    		double delta[] = new double[trainingSet.get(0).classValues.size()];
	    		//Determining the denominator for the output of the output node.
	    		for (int k = 0; k < trainingSet.get(0).classValues.size(); k++) {
	    			output[k] =  Math.exp(outputNodes.get(k).getOutput()); 
	    			denominator =  denominator + output[k];
	    		}
	    		//Determining the output values, delta values, and updating weight for an output node.
	    		for (int k = 0; k < trainingSet.get(0).classValues.size(); k++) {
	    			output[k] = output[k]/denominator;
	    			//Determining the delta of output node.
	    			delta_j = outputNodes.get(k).calculateDelta(trainingSet.get(i).classValues, output[k], outputNodes, delta, k, outputWeights_copy);
	    			delta[k] = delta_j;
	    			//Updating the weight between hidden and output layer.
	    			outputNodes.get(k).updateWeight(learningRate, delta_j);
	    		}
	    		//Updating the weights of the hidden node.
	    		for (int k = 0; k < hiddenNodeCounter; k++) {
	    			//Determining the delta of input node.
	    			delta_j = hiddenNodes.get(k).calculateDelta(trainingSet.get(i).classValues, hiddenNodes.get(k).getOutput(), outputNodes, delta, k, outputWeights_copy);
	    			//Updating the weight between input and hidden layer.
	    			hiddenNodes.get(k).updateWeight(learningRate, delta_j);
	    		}
	    	}
	     	//Determining the cross-entropy loss.
	     	for (int k = 0; k < trainingSet.size(); k++) {
	     		sum_epoch = sum_epoch + loss(trainingSet.get(k));
	     	}	
	     	//Printing the cross-entropy loss.
	     	System.out.format("Epoch: %s, Loss: %.8e\n", e, sum_epoch/trainingSet.size());
    	}
    }
	/**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
    	double loss_entropy, denominator;
    	loss_entropy = denominator = 0;
    	double output_loss[] = new double[trainingSet.get(0).classValues.size()];
		for (int j = 0; j < trainingSet.get(0).attributes.size(); j++) {//Assigning value of each attribute to the node.
			inputNodes.get(j).setInput(instance.attributes.get(j));
		}
    	for (int k = 0; k < trainingSet.get(0).classValues.size(); k++) {//Determining the output.
    		output_loss[k] =  Math.exp(outputNodes.get(k).getOutput()); 
    		denominator =  denominator + output_loss[k];
    	}
    	//Determining the cross-entropy loss.
    	for (int k = 0; k < trainingSet.get(0).classValues.size(); k++) {
    		output_loss[k] = output_loss[k]/denominator;
    		loss_entropy =  loss_entropy - instance.classValues.get(k)*Math.log(output_loss[k]);
    	}
        return loss_entropy;
    }
}