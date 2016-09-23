package sim.velocity.obstacles;

import java.util.ArrayList;
import java.util.TreeMap;

import math.geom2d.Point2D;
import math.geom2d.Vector2D;
import math.geom2d.conic.Circle2D;
import math.geom2d.line.Ray2D;
import sim.elements.Bot;
import sim.elements.Thing;
import sim.geometry.Trapezoid;
import sim.graphics.World;
import sim.structures.Vector;

public class AdmissableVelocity {
	private ArrayList<VelocityObstacle> vos;
	private ArrayList<Vector> available;
	private TreeMap<Double,Vector> velocities;
	private Bot self;
	private boolean collision;
	boolean test=true;
	
	private int sampling;
	public static final int SAMPLING_ANGULAR=0;
	public static final int SAMPLING_GRID=1;
	public static final int SAMPLING_RANDOM=3;

	public int selecting;
	public static final int SELECT_COLLISION_FREE=0;
	public static final int SELECT_TIME_TO_COLLISION=1;
	public static final int SELECT_COMBINED_TIME_TO_COLLISION=2;
	
	public int velocity_obstacle;
	public static final int VELOCITY_OBSTACLE=0;
	public static final int RECIPROCAL_VELOCITY_OBSTACLE=1;
	public static final int AUGMENTED_RECIPROCAL_VELOCITY_OBSTACLE=2;
	
	
	private Vector admissableVelocity=null;
	boolean augmented;
	public static boolean prune_search_space = false;
	
	public AdmissableVelocity(Bot self,int sampling,int selecting,int velocity_obstacle){
		this.self = self;
		this.vos = new ArrayList<VelocityObstacle>();
		this.sampling = sampling;
		this.selecting = selecting;
		this.collision = true;
		this.velocity_obstacle = velocity_obstacle;
		this.augmented = this.velocity_obstacle == AdmissableVelocity.AUGMENTED_RECIPROCAL_VELOCITY_OBSTACLE;
	}
	
	public boolean getCollision(){
		return this.collision;
	}
	
	public void createVelocityObstacle(Thing obstacle){
		VelocityObstacle vo = new VelocityObstacle(this.self,obstacle);
		vo.createCollisionCone(this.augmented);//CREATE COLLISION CONE
		
		if(collision=vo.collision()){
		//if(!(collisionFree=vo.isCollisionFree(this.augmented))){//CHECK IF RELATIVE VELOCITY IS IN COLLISION CONE
			//if(this.velocity_obstacle == AdmissableVelocity.VELOCITY_OBSTACLE) vo.createVelocityObstacle(obstacle.getV());//CREATE VELOCITY OBSTACLE
			//else if(this.velocity_obstacle == AdmissableVelocity.RECIPROCAL_VELOCITY_OBSTACLE) vo.createReciprocalVelocityObstacle(self.getV(),obstacle.getV());
			this.vos.add(vo);
			if(World.drawVelocityObstacle) World.velocity_obstacle(vo);
		}
	}
	
	public void extractVelocities(Vector location,Vector preferredVelocity){
		if(AdmissableVelocity.SAMPLING_ANGULAR == this.sampling){//CHOOSE VELOCITY GENERATOR
			this.available = (new VelocityExtractor(location,preferredVelocity,self.getS())).extractAngularVelocities(10, 360);
		}else if(AdmissableVelocity.SAMPLING_GRID == this.sampling){
			this.available = (new VelocityExtractor(location,preferredVelocity,self.getS())).extractGridVelocities((int)(-self.getS()), (int)(self.getS()));
		}else if(AdmissableVelocity.SAMPLING_RANDOM == this.sampling){
			this.available = (new VelocityExtractor(location,preferredVelocity,self.getS())).extractRandom(250,(int)self.getMaxSpeed(),self.seed);
		}
		
		if(AdmissableVelocity.SELECT_COLLISION_FREE == this.selecting) this.onlyCollisionFree(preferredVelocity);
		else if(AdmissableVelocity.SELECT_TIME_TO_COLLISION == this.selecting) this.calculatePenalty(preferredVelocity, 20);
	}
	
	public Vector newVelocity(){
		return this.admissableVelocity;
	}
	
	public void onlyCollisionFree(Vector preferredVelocity){
		this.velocities = new TreeMap<Double,Vector>();
		
		boolean  collision=false;
		double penalty=0;
		for(Vector velocity : this.available){
			collision=false;
			penalty=0;
			for(VelocityObstacle vo : this.vos){
				if(collision=!vo.isCollisionFree(velocity,this.augmented)) break;
			}
			if(!collision){
				if(!this.velocities.isEmpty() && penalty> this.velocities.firstKey()) break;
				penalty = (new Vector(velocity,preferredVelocity)).norm();
				this.velocities.put(penalty, velocity);
			}
		}
	}
	
	public void calculatePenalty(Vector preferredVelocity,double weight){
		double penalty=0,minPenalty=Double.MAX_VALUE,minTime=Double.MAX_VALUE,tmp=0;
		int i=0;
		
		for(Vector velocity : this.available){
			penalty=0;
			minTime=Double.MAX_VALUE;
			for(VelocityObstacle vo : this.vos){
				//if(vo.collision(this.getRelativeVelocity(self.getV(), vo.getObstacle().getV(), preferredVelocity))){
				if(!vo.isCollisionFree(velocity,this.augmented)){
					tmp = this.timeToCollision(self.getCenter(), 
							vo.getObstacle().getCenter(),
							velocity, self.getS(), 
							vo.getObstacle().getS(),vo.getTrapezoid());
					if(minTime>tmp) minTime = tmp;
					if(penalty<0) System.out.println("Negative");
				}
			}
			i++;
			penalty = weight / minTime + (new Vector(velocity,preferredVelocity)).norm();
			if(minPenalty>penalty){ minPenalty = penalty; this.admissableVelocity = velocity; }
			else if(prune_search_space && i>=100 && this.admissableVelocity!=null){ break; }
		}
	}
	
	private Double timeToCollision(Vector cA,Vector cB,Vector velocity,double selfSize, double obstacleSize,Trapezoid t){
		Ray2D ray = new Ray2D(cA,new Vector2D(velocity));
		Circle2D circleB = new Circle2D(cB,obstacleSize+selfSize);
		ArrayList<Point2D> points = new ArrayList<Point2D>();
		points.addAll(circleB.intersections(ray));
		
		if(points.size()==0) return Double.MAX_VALUE;
		else if (points.size()==1) return Double.MIN_VALUE;
		else return this.minTime(points, ray);
	}
	
	private Double minTime(ArrayList<Point2D> points, Ray2D ray){
		Double t = Double.MAX_VALUE;
		
		for(Point2D point : points){
			if(ray.positionOnLine(point)<t){
				t = ray.positionOnLine(point);
			}
		}
		
		return t;
	}
}
