package sim.collision.avoidance.planned.squared;

import java.awt.Color;

import sim.elements.Bot;
import sim.structures.Vector;

public class PlannedBot extends Bot {
	int id;
	Planned p;
	boolean init=true;
	
	public PlannedBot(double size, Color color, int id,Planned p) {
		super(size, color);
		this.id = id;
		this.p = p;
	}

	@Override
	public void act() {
		if(init){
			init=false;
		}
		
		if(!p.getMap().get(id).isEmpty()){
			Vector at = p.getMap().get(id).get(0);
			Vector go = new Vector(this.loc,at);
			this.velocity.set(go.getX(), go.getY());
			p.getMap().get(id).remove(0);
		}else{
			this.velocity.set(0, 0);
		}
		
	}

}
