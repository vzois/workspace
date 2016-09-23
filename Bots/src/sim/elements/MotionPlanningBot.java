package sim.elements;

import java.awt.Color;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public abstract class MotionPlanningBot extends Bot{
	private Vector target;
	private Vector preferredVelocity;
	protected int sampling,selection,velocity_obstacle;	
	
	public MotionPlanningBot(double size, double range,Color color){
		super(size,range,color);
	}
	
	public void setTarget(Vector target){
		this.target = new Vector(target);
	}
	
	public Vector getTarget(){
		return this.target;
	}
	
	public void setPreferredVelocity(Vector preferredVelocity){
		this.preferredVelocity = new Vector(preferredVelocity);
	}

	public Vector planMotion(){
		this.velocity.set(this.preferredVelocity.getX(), this.preferredVelocity.getY());
		Vector oldVelocity = new Vector(this.preferredVelocity);
		Vector newVelocity = null;
		AdmissableVelocity av = new AdmissableVelocity(this,sampling,selection,velocity_obstacle);
		for(Object o : this.getNeighbors()){
			if(!o.equals(this)) av.createVelocityObstacle((Thing)o);
		}
		if(av.getCollision()){
			av.extractVelocities(this.getCenter(),oldVelocity);
			newVelocity= av.newVelocity();
		}
		
		if(!av.getCollision() && newVelocity!=null && oldVelocity.contains(newVelocity)){
			System.out.println("Not Admissible Velocity Found ("+oldVelocity+") ("+newVelocity+")");
		}
		return newVelocity!=null ? newVelocity : oldVelocity;
	}
}
