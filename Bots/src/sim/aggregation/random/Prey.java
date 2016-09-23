package sim.aggregation.random;
import java.awt.Color;

import sim.elements.Thing;
import sim.structures.Vector;



public class Prey extends Thing{
	protected Vector loc;
	public Prey(int size){
		super(size,Color.GREEN);
	}
}
