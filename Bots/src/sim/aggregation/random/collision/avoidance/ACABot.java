package sim.aggregation.random.collision.avoidance;

import java.awt.Color;

import sim.elements.MotionPlanningBot;
import sim.elements.Thing;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public class ACABot extends MotionPlanningBot {
	boolean prey = false;
	private int time;
	private int bias;
	Thing t=null;
	Color c1 = Color.CYAN;
	Color c2 = Color.BLUE;
	boolean aggregate = false;
	
	public ACABot(double size,double range,int bias){
		super(size,range,Color.BLUE);
		this.bias=bias;
		this.setMaxSpeed(5);
		sampling = AdmissableVelocity.SAMPLING_RANDOM;
		selection = AdmissableVelocity.SELECT_TIME_TO_COLLISION;
		velocity_obstacle = AdmissableVelocity.AUGMENTED_RECIPROCAL_VELOCITY_OBSTACLE;
		time=0;
	}
	
	public void act(){
		
		if(!prey){
			Object o = null;
			for(int i = 0;i<this.getNeighbors().length;i++){
				o = this.getNeighbors()[i];
				if(((Thing)o).getC()==Color.GREEN && !o.equals(this)){
					prey=true;
					t = (Thing)o;
					break;
				}
			}
		}
		
		if(prey){
				velocity.points(this.getCenter(),t.getCenter());
				this.limitSpeed();
				color = Color.YELLOW;
		}else if(time%bias==0){
			this.velocity.randD(this.getMaxSpeed(), seed);
		}
		
		if(!(this.getC()==Color.RED)){
			this.setTarget(Vector.add(this.getCenter(),this.velocity));
			selectVelocity();
		}
		
		time++;
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
