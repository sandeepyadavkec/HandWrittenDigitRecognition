/**
 * 
 */
package neural;

import java.util.ArrayList;
import java.util.List;

/**
 * @author SandeepYadav
 *
 */
public class Net {
	
	private double m_recentAverageSmoothingFactor = 100.0;
	private double m_recentAverageError = 0.0;
	
	List<List<Neuron>> m_layers = new ArrayList<List<Neuron>>();
	
	public Net(List<Integer> topology){
		
		int numLayers = topology.size();
						
		for(int layerNum=0; layerNum < numLayers; layerNum++) {
			List<Neuron> Layer = new ArrayList<Neuron>();							
			int numOutputs = (layerNum == numLayers-1) ? 0 : topology.get(layerNum+1);
			for(int neuronNum=0; neuronNum <= topology.get(layerNum); neuronNum++) {
				Neuron neuron = new Neuron(numOutputs, neuronNum);
				Layer.add(neuron);
				
			}
			m_layers.add(Layer);
			m_layers.get(layerNum).get(topology.get(layerNum)).setOutputVal(1.0);
		}
		
	}
	
	public void feedForward(List<Double> inputVals) {
		// Feed forward
		assert(inputVals.size()==m_layers.get(0).size()-1);
		
		//Latch inputs to the input layer
		for(int i = 0; i < inputVals.size(); i++) {
			m_layers.get(0).get(i).setOutputVal(inputVals.get(i));			
		}
		
		//Aggregated sum
		for(int layerNum=1; layerNum < m_layers.size(); layerNum++) {
			List<Neuron> prevLayer = m_layers.get(layerNum-1);
			for(int n=0; n < m_layers.get(layerNum).size()-1; n++) {
				System.out.println("Layer: "+layerNum);
				m_layers.get(layerNum).get(n).feedForward(prevLayer);
			}
		}
		
	}

	public void backProp(List<Double> targetOutputVals) {
		// backwards propagation
		
		//Calculate overall net error (RMS)
		List<Neuron> outputLayer = m_layers.get(m_layers.size()-1);
		double m_error = 0.0;
		
		for (int neuronNum=0; neuronNum < outputLayer.size()-1; neuronNum++) {
			System.err.print("Output layer neuron num: "+neuronNum);
			System.err.print("\tTarget value: "+targetOutputVals.get(neuronNum).doubleValue());
			System.err.print("\tResult value: "+outputLayer.get(neuronNum).getOutputVal()+"\n");
			double delta = targetOutputVals.get(neuronNum).doubleValue() - outputLayer.get(neuronNum).getOutputVal();
			m_error += (delta * delta); //get sum of squares of delta
		}
		
		m_error = m_error/(outputLayer.size()-1); //get average of squares
		m_error = Math.sqrt(m_error);
		if(m_error>0.09){
			System.err.println("Err: "+m_error);
		}
		
		//Implement recent average measurement		
		m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);
		System.err.println("Avg Err: "+m_recentAverageError);
		
		//Calculate output layer gradients		
		for (int neuronNum=0; neuronNum < outputLayer.size()-1; neuronNum++) {
			outputLayer.get(neuronNum).calcOutputGradients(targetOutputVals.get(neuronNum));
		}
		
		//Calculate hidden layer gradients		
		for(int layerNum=m_layers.size()-2; layerNum >0; layerNum--) {
			List<Neuron> hiddenLayer = m_layers.get(layerNum);
			List<Neuron> nextLayer = m_layers.get(layerNum+1);
			for (int neuronNum=0; neuronNum < hiddenLayer.size(); neuronNum++) {
				hiddenLayer.get(neuronNum).calcHiddenGradients(nextLayer);
			}
		}
		
		//For all layers from output to first hidden layer, update connection weights		
		for(int layerNum=m_layers.size()-1; layerNum >0; layerNum--) {			
			List<Neuron> Layer = m_layers.get(layerNum);
			List<Neuron> prevLayer = m_layers.get(layerNum-1);
			for(int n=0; n < Layer.size()-1; n++) {
				Layer.get(n).updateInputWeights(prevLayer);
			}
		}
		
	}

	public List<Double> getResults() {
		// Populate results in output layer
		List<Double> resultVals = new ArrayList<Double>();
		
		List<Neuron> outputLayer = m_layers.get(m_layers.size()-1);
		
		for (int neuronNum=0; neuronNum < outputLayer.size()-1; neuronNum++) {
			resultVals.add(neuronNum, outputLayer.get(neuronNum).getOutputVal());
		}
		
		return resultVals;
	}
	
	public void dumpResults() {
		//For debugging purposes
		for(int layerNum=m_layers.size()-1;layerNum<m_layers.size();layerNum++) {
			for(int neuronNum=0;neuronNum<m_layers.get(layerNum).size();neuronNum++){
				System.out.println("***********************************************************");
				System.out.println(">>>>Layer: "+layerNum+"\tNeuron: "+neuronNum);
				System.out.println("output: "+m_layers.get(layerNum).get(neuronNum).getOutputVal());
				System.out.println("gradient: "+m_layers.get(layerNum).get(neuronNum).m_gradient);
				System.out.println("my index: "+m_layers.get(layerNum).get(neuronNum).m_myIndex);
				System.out.println("delta weights size: "+m_layers.get(layerNum).get(neuronNum).m_deltaWeights.size());
				System.out.println("output weights size: "+m_layers.get(layerNum).get(neuronNum).m_outputWeights.size());
				
			}
		}
	}
	

}
