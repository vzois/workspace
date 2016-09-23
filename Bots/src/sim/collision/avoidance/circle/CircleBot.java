package sim.collision.avoidance.circle;

import java.awt.Color;

import sim.elements.MotionPlanningBot;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public class CircleBot extends MotionPlanningBot{		
	public CircleBot(Color c){
		super(10, 25,c);
		this.setMaxSpeed(5);
		sampling = AdmissableVelocity.SAMPLING_RANDOM;
		selection = AdmissableVelocity.SELECT_TIME_TO_COLLISION;
		velocity_obstacle = AdmissableVelocity.AUGMENTED_RECIPROCAL_VELOCITY_OBSTACLE;
	}
	

	@Override
	public void act() {
		if(this.getCenter().distance(this.getTarget())>this.getS()){
			selectVelocity();
		}else{
			this.setL(this.getTarget());
			this.stop();
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