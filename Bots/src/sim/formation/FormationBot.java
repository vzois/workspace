package sim.formation;

import java.awt.Color;

import sim.elements.Bot;
import sim.elements.MotionPlanningBot;
import sim.geometry.FormationShape;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

public class FormationBot extends MotionPlanningBot {
	boolean init = true;
	boolean leader = true;
	FormationShape shape;
	int shapeChoice;
	
	public static final int LEADER_ELECT=0;
	public static final int SHAPE_BROADCAST=1;
	public static final int SHAPE_READJUST=2;
	
	public static final int WAIT_FOR_POS=0;
	public static final int MOVE_TO_POS=1;
	
	int state = 0;
	int id=0;
	int agents =0;
	int test=0;
	
	public FormationBot(boolean leader,int shapeChoice,int id) {
		super(10,1000, Color.CYAN);
		this.setMaxSpeed(5);
		//this.leader = leader;
		this.shapeChoice = shapeChoice;
		this.id = id;
		sampling = AdmissableVelocity.SAMPLING_RANDOM;
		selection = AdmissableVelocity.SELECT_TIME_TO_COLLISION;
		velocity_obstacle = AdmissableVelocity.AUGMENTED_RECIPROCAL_VELOCITY_OBSTACLE;
	}

	@Override
	public void act() {
		if(init){
			electLeader();
			//System.out.println("id: "+id+" = election in progress!");
		}else if(test==0){
			test++;
			this.range = 25;
		}else if(leader && !init){
			if(FormationBot.LEADER_ELECT == state){
				this.setC(Color.RED);
				this.setTarget(new Vector(350,350));
				if(this.getCenter().distance(this.getTarget())>this.getS()){
					selectVelocity();
				}else{
					System.out.println(id+" :leader: "+leader);
					this.stop();
					this.range = 1000;
					this.setL(this.getTarget());
					state++;
				}
			}else if(FormationBot.SHAPE_BROADCAST == state){
				this.setC(Color.ORANGE);
				this.agents = this.getNeighbors().length;
				this.selectShape(this.agents);
				for(Object t : this.getNeighbors()){
					if(t instanceof Bot && !t.equals(this)){
						Bot b = (Bot)t;
						if(!b.equals(this)){
							ShapeMessage sm = new ShapeMessage();
							sm.setShape(this.shape.getShape());
							b.writeMessageQueue(sm);
						}
					}
				}
				this.range = 25;
				state++;
			}else if(FormationBot.SHAPE_READJUST == state){
				this.setTarget(new Vector(this.shape.getShape().get(id)));
				if(this.getCenter().distance(this.getTarget())>this.getS()){
					selectVelocity();
					this.setC(Color.ORANGE);
				}else{
					this.stop();
					this.setL(this.getTarget());
					this.setC(Color.GREEN);
				}
			}
		}else if(!leader && !init){
			if(FormationBot.WAIT_FOR_POS == state){
				ShapeMessage sm = (ShapeMessage) this.readMessageQueue();
				if(sm!=null){
					Vector v = new Vector(sm.getShape().get(id));
					this.setTarget(v);
					state++;
				}
			}else if(FormationBot.MOVE_TO_POS == state){
				this.setC(Color.ORANGE);
				if(this.getCenter().distance(this.getTarget())>this.getS()){
					selectVelocity();
				}else{
					this.stop();
					state++;
				}
			}else{
				this.setC(Color.GREEN);
			}
		}
	}

	private void electLeader(){
		int weight = (int) this.seed.nextInt(Integer.MAX_VALUE);
		this.agents = this.getNeighbors().length-1;
		for(Object t : this.getNeighbors()){
			if(t instanceof Bot && !t.equals(this)){
				Bot b = (Bot)t;
				b.writeMessageQueue(new ElectMessage(this,weight));
			}
		}
		
		while(this.mq.size() != this.agents){this.sleep(1000);}
		init = false;
		
		ElectMessage em;
		while((em = (ElectMessage)this.readMessageQueue())!=null){
			if(em.getWeight()>weight){leader = false; }
			else if(em.getWeight()==weight){ init =true; leader=true; }
		}
		
		this.mq.clear();
	}

	private void selectShape(int agents){
		if(shapeChoice == FormationShape.CIRCLE){
			shape = new FormationShape(agents);
			shape.createCircle(this.getCenter(), 200);
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
