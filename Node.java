import java.util.*;
/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */
public class Node {
    private int type = 0; //0=input, 1=biasToHidden, 2=hidden, 3=biasToOutput, 4=Output.
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }
    //For an input node, sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {//If input node
            this.inputValue = inputValue;
        }
    }
    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {    	
    	double input_to_layer;
        if (type == 2 || type == 4) {//Not an input or bias node
        	if (type == 2) {//Calculating output for the hidden node.
        		input_to_layer = 0.0;
                for (int i = 0; i < parents.size()-1; i++) {
                	input_to_layer = input_to_layer + (parents.get(i).weight)*parents.get(i).node.inputValue;
                }
                input_to_layer = input_to_layer + (parents.get(parents.size()-1).weight);
                outputValue = Math.max(0, input_to_layer);
        	}
        	else {//Calculating output for the output node.
        		input_to_layer = 0.0;
                for (int i = 0; i < parents.size()-1; i++) {
                	input_to_layer = input_to_layer + (parents.get(i).weight)*parents.get(i).node.getOutput();
                }
                input_to_layer = input_to_layer + (parents.get(parents.size()-1).weight);
        		outputValue = input_to_layer;
        	}
        }
    }
    //Gets the output value
    public double getOutput() {    	
        if (type == 0) {//Input node
            return inputValue;
        } else if (type == 1 || type == 3) {//Bias node
            return 1.00;
        } else {
        	calculateOutput();
            return outputValue;
        }
    }
    //Calculate the delta value of a node.
    public double calculateDelta(ArrayList<Integer> teacher, double z, ArrayList<Node> outputNodes, double delta_output[], int node_number, Double [][] output_weight) {
        if (type == 2 || type == 4)  {
            if (type == 2) {
            	double z_prime = 0;
            	double sum = 0.0;
            	//Determining the gradient.
            	if (z > 0) {
            		z_prime = 1;
            	}
            	//Determining the delta for the hidden node.
            	for (int k = 0; k < outputNodes.size(); k++) {
            		sum = sum + output_weight[k][node_number]*delta_output[k];
            	}
            	delta = z_prime*sum;           	
            }
            else {//Delta for output node.
            	delta = teacher.get(node_number) - z;
            }
        }
        return delta;
    }
    //Update the weights between parents node and current node
    public void updateWeight(double learningRate, double delta_j) {
        if (type == 2 || type == 4) {
    		for (int i = 0; i < parents.size(); i++) {
    			//Determining the gradient.
    			outputGradient = learningRate*parents.get(i).node.getOutput()*delta_j;
    			//Updating the weights.
    			parents.get(i).weight = parents.get(i).weight + outputGradient;
    		}
        }
    }
}