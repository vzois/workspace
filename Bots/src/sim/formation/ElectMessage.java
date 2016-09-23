package sim.formation;

import sim.elements.Bot;
import sim.structures.Message;

public class ElectMessage extends Message{
	int weight;
	
	public ElectMessage(Bot b,int weight){
		super(b,new Long(0),-1);
		this.weight = weight;
	}
	
	public int getWeight(){
		return this.weight;
	}

}
