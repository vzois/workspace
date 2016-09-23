package sim.collision.avoidance.lineofrobots;

import java.awt.Color;

import sim.elements.MotionPlanningBot;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public class TransparentBot extends MotionPlanningBot{
	private Vector targetA;
	private Vector targetB;
	private boolean init = true;
	
	public TransparentBot(Color color) {
		super(10, 100, color);
		this.setMaxSpeed(5);
		sampling = AdmissableVelocity.SAMPLING_RANDOM;
		selection = AdmissableVelocity.SELECT_TIME_TO_COLLISION;
		velocity_obstacle = AdmissableVelocity.RECIPROCAL_VELOCITY_OBSTACLE;
	}

	@Override
	public void act() {
		if(init){
			targetA = new Vector(this.getCenter());
			targetB = new Vector(this.getCenter());
			if(this.getC() == Color.YELLOW) targetB.add(new Vector(0,550));
			else targetB.add(new Vector(0,-550));
			init = false;
			this.setTarget(targetA);
		}
		
		if(this.getCenter().distance(this.getTarget())>2*this.size){
			selectVelocity();
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
