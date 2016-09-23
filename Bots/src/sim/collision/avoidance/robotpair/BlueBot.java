package sim.collision.avoidance.robotpair;

import java.awt.Color;

import sim.elements.MotionPlanningBot;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public class BlueBot extends MotionPlanningBot{
	private boolean init=false;
	private Vector targetA= new Vector(300,50);
	private Vector targetB= new Vector(300,550);
	
	public BlueBot(double size, double range) {
		super(size, range, Color.BLUE);
		this.setMaxSpeed(5);
		sampling = AdmissableVelocity.SAMPLING_RANDOM;
		selection = AdmissableVelocity.SELECT_TIME_TO_COLLISION;
		velocity_obstacle = AdmissableVelocity.AUGMENTED_RECIPROCAL_VELOCITY_OBSTACLE;
	}
	
	@Override
	public void act() {
		if(!init){ this.setTarget(targetA); init=true; }
		if(this.loc.distance(this.getTarget())>2*this.size){
			selectVelocity();
			//System.out.println("final: "+this.velocity);
		}else{
			if(this.getTarget().contains(targetA)){
				this.setTarget(targetB);
			}else{
				this.setTarget(targetA);
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
