package sim.aggregation.follow;

import sim.elements.Bot;
import sim.structures.Message;
import sim.structures.Vector;

public class FormationMessage extends Message {
	private Vector pos;
	public FormationMessage(Bot bot, Long timestamp,Vector pos) {
		super(bot, timestamp);
		this.pos = pos;
	}
	
	public Vector getPos(){
		return this.pos;
	}

}
