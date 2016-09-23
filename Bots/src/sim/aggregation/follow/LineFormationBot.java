package sim.aggregation.follow;

import java.awt.Color;
import java.util.ArrayList;

import sim.elements.Bot;
import sim.elements.MotionPlanningBot;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public class LineFormationBot extends MotionPlanningBot {
	private boolean init = true;
	private boolean leader = false;
	private int state = 0;
	private ArrayList<Vector> formation;
	
	private static final int MOVE=0;
	private static final int TRANSMIT=1;
	private static final int SEARCH = 2;
	
	public LineFormationBot(boolean leader) {
		super(10, 15, Color.CYAN);
		this.leader = leader;
		this.setMaxSpeed(5);
		sampling = AdmissableVelocity.SAMPLING_RANDOM;
		selection = AdmissableVelocity.SELECT_TIME_TO_COLLISION;
		velocity_obstacle = AdmissableVelocity.RECIPROCAL_VELOCITY_OBSTACLE;
	}

	@Override
	public void act() {
		if(init){
			init = false;
			if(leader){
				this.setC(Color.RED);
				this.formation = new ArrayList<Vector>();
				this.setTarget(new Vector(300,300));
			}else{
				state = LineFormationBot.SEARCH;
			}
		}
		
		if(leader){
			if(state == LineFormationBot.MOVE){
				if(this.getCenter().distance(this.getTarget())>this.getS()){
					selectVelocity();
				}else{
					this.velocity.set(0, 0);
					this.range = 1000;
					state++;
				}
			}else if(state == LineFormationBot.TRANSMIT){
				this.setC(Color.ORANGE);
				for(Object t : this.getNeighbors()){
					Vector relativePos;
					if(t instanceof Bot){
						if(this.formation.isEmpty()){
							relativePos = new Vector(this.getCenter());
							relativePos.add(new Vector(2*this.getS(),0));
							((Bot) t).writeMessageQueue(new FormationMessage(this,new Long(0),relativePos));
							this.formation.add(0, relativePos);
						}else{
							relativePos = new Vector(this.formation.get(0));
							relativePos.add(new Vector(2*this.getS(),0));
							((Bot) t).writeMessageQueue(new FormationMessage(this,new Long(0),relativePos));
							this.formation.add(0, relativePos);
						}
					}
				}
				state++;
			}else{
				this.setC(Color.GREEN);
			}
		}else{
			if(state == LineFormationBot.SEARCH){
				FormationMessage fm = (FormationMessage)this.readMessageQueue();
				if(fm!=null){
					this.setTarget(fm.getPos());
					this.setC(Color.BLUE);
					state= LineFormationBot.MOVE;
				}
			}else if(state == LineFormationBot.MOVE){
				if(Math.floor(this.getL().distance(this.getTarget())-2)>this.getS()){
					selectVelocity();
				}else{
					this.setL(this.getTarget());
					this.velocity.set(0, 0);
				}
			}
		}
		
	}
	
	private void selectVelocity(){
		Vector vi = new Vector(this.getCenter(),this.getTarget());
		vi.normalize();
		vi.multi(this.getMaxSpeed());
		this.setPreferredVelocity(vi);
		vi = this.planMotion();
		this.velocity = new Vector(vi);
		this.limitSpeed();
	}

}
