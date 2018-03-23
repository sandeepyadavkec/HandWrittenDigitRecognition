package neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.lang.Math;

public class Neuron {
	
	List<Double> m_outputWeights = new ArrayList<Double>();
	List<Double> m_deltaWeights = new ArrayList<Double>();
	private double m_outputVal;
	Random rand = new Random();
	int m_myIndex;
	double m_gradient;
	static double eta = 0.15;
	static double alpha = 0.5;
	
	public Neuron(int numOutputs, int myIndex){
		
		//Initialize output and delta weights
		for(int c=0; c < numOutputs; c++) {
			m_outputWeights.add(c, rand.nextDouble());
			System.out.println("Random output weight: "+m_outputWeights.get(c));
			m_deltaWeights.add(0.0);			
		}
		m_myIndex = myIndex;		
	}

	public void setOutputVal(double val) {
		// set output values
		m_outputVal = val;
	}
	
	public double getOutputVal() {
		// get output values
		return m_outputVal;
	}

	public void feedForward(List<Neuron> prevLayer) {
		// aggregated summation
		double sum = 0.0;
		
		for(int n=0; n<prevLayer.size(); n++) {
			sum += prevLayer.get(n).getOutputVal() * 
			prevLayer.get(n).m_outputWeights.get(m_myIndex);
			//System.out.println("prev layer neruon: "+n);
		}
		m_outputVal=transferFunction(sum/prevLayer.size()-1);
		System.out.println("output val: "+m_outputVal);
	}

	private double transferFunction(double x) {
		
		//return Math.tanh(x);
		return 1 / (1 + Math.exp(-x));
	}
	
	private double transferFunctionDerivative(double x) {
		// tanh derivative
		return (1.0 - (x * x));		
	}

	public void calcOutputGradients(Double targetVal) {
		System.out.print("Calc output gradient: ");
		double delta = targetVal - m_outputVal;
		System.out.println("delta, target, output: "+delta+"\t"+targetVal+"\t"+m_outputVal);
		m_gradient = delta * transferFunctionDerivative(m_outputVal);
		System.out.println("gradient = delta * (1-output*output) = "+m_gradient);
	}

	public void calcHiddenGradients(List<Neuron> nextLayer) {		
		double dow = sumDOW(nextLayer);
		m_gradient = dow * transferFunctionDerivative(m_outputVal);
		}

	private double sumDOW(List<Neuron> nextLayer) {		
		double sum = 0.0;		
		for(int n=0; n < nextLayer.size()-1; n++) {
			sum += m_outputWeights.get(n).doubleValue() * nextLayer.get(n).m_gradient;
		}
		return sum;
	}

	public void updateInputWeights(List<Neuron> prevLayer) {		
		for(int n=0; n < prevLayer.size(); n++) {
			Neuron neuron = prevLayer.get(n);
			double oldDeltaWeight = neuron.m_deltaWeights.get(m_myIndex).doubleValue();
			
			//Individual input, magnified by the gradient and the training rate
			double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
			
			neuron.m_deltaWeights.set(m_myIndex, newDeltaWeight);
			neuron.m_outputWeights.set(m_myIndex, neuron.m_outputWeights.get(m_myIndex)+newDeltaWeight);
		}
		
	}

}
