package sim.collision.avoidance.swarm;

import java.awt.Color;

import sim.elements.MotionPlanningBot;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public class ColorFullBot extends MotionPlanningBot{
	private boolean init=true;
	
	
	public ColorFullBot(Color c){
		super(10, 25,c);
		this.setMaxSpeed(5);
		sampling = AdmissableVelocity.SAMPLING_RANDOM;
		selection = AdmissableVelocity.SELECT_TIME_TO_COLLISION;
		velocity_obstacle = AdmissableVelocity.AUGMENTED_RECIPROCAL_VELOCITY_OBSTACLE;
	}

	@Override
	public void act() {
		if(init){ this.velocity.randD(this.getMaxSpeed(), seed); init = false; }		
		if(this.velocity.norm()<=2){ this.velocity.multi(2); }
		
		this.setTarget(Vector.add(this.getCenter(), this.velocity));
		selectVelocity();
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
