package test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import neural.Net;

public class Test {

	private static List<Double> inputVals = new ArrayList<Double>();
	private static List<Double> targetVals = new ArrayList<Double>();
		
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
				
		TestReadImageDigit();
		
	}
	
	private static void TestReadImageDigit(){
		List<Double> resultVals = null;
		List<Integer> topology = new ArrayList<Integer>();
					
		topology.add(784);
		topology.add(16);
		topology.add(16);
		topology.add(10);
		Net myNet = new Net(topology);
		
		List<String[]> contentStr = readCSVRow();
		
		for(int i=0;i<contentStr.size();i++){
			System.out.println("Pass: "+i);
			inputVals.clear();
			targetVals.clear();
	    	Double[] temp=new Double[785];
	    	
	    	for(int j=1;j<contentStr.get(i).length;j++){
	    		temp[j] =Double.parseDouble(contentStr.get(i)[j]);
	    		inputVals.add(temp[j]/255);
	    		//System.out.println(j+".\t"+temp[j]/255);
	    	}
	    	
	    	temp[0] =Double.parseDouble(contentStr.get(i)[0]);
	    	for(int k=0;k<=9;k++){
	    		if(k==(int)temp[0].doubleValue()){
	    			targetVals.add(1.0);
	    		}
	    		else{
	    			targetVals.add(0.0);
	    		}
	    	}
	    	
	    	
			//Pass inputs
			myNet.feedForward(inputVals);
			
			//Get results
			resultVals = myNet.getResults();
			
								
			//Train the net
			myNet.backProp(targetVals);
			
			//for debugging
			//myNet.dumpResults();
	    }
	}
	
	private static List<String[]> readCSVRow(){
		
		String file = "mnist_train.csv";
	    List<String[]> contentStr = new ArrayList<String[]>();
	    
	    try{
	    	BufferedReader br = new BufferedReader(new FileReader(file));
	        String line = "";
	        try {
				while ((line = br.readLine()) != null) {
					contentStr.add(line.split(","));
					System.out.print(".");
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	    } catch (FileNotFoundException e) {
	      //Some error logging
	    }
	    return contentStr;
	    
	}
			

}
