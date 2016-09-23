package sim.velocity.obstacles;

import java.util.ArrayList;

import math.geom2d.Point2D;
import math.geom2d.conic.Circle2D;
import math.geom2d.line.Ray2D;
import sim.elements.Bot;
import sim.elements.Thing;
import sim.geometry.CircularSector;
import sim.geometry.Trapezoid;
import sim.structures.Vector;

public class VelocityObstacle {
	protected Bot self;
	protected Thing obstacle;
	protected CircularSector cs;
	protected Trapezoid trapezoid;
	protected boolean collisionFree;
	
	public VelocityObstacle(Bot self,Thing obstacle){
		this.self = self;
		this.obstacle = obstacle;
		this.collisionFree = false;
	}
	
	public CircularSector getSector(){
		return this.cs;
	}
	
	public Trapezoid getTrapezoid(){
		return this.trapezoid;
	}
	
	public void createCollisionCone(boolean augmented){
		double collisionObjectRadius = ((this.self.getS()/2 + obstacle.getS()/2));
		double radiusOfCollisionCone = this.self.getCenter().distance(obstacle.getCenter());
		radiusOfCollisionCone = Math.sqrt(radiusOfCollisionCone*radiusOfCollisionCone - collisionObjectRadius*collisionObjectRadius);
		Circle2D obstacleCircle = new Circle2D(obstacle.getCenter(),collisionObjectRadius);//CREATE CIRCLE WITH RADIUS SUM OF OBSTACLE RADIUS + SELF RADIUS 
		Circle2D collisionConeCircle = new Circle2D(this.self.getCenter(),radiusOfCollisionCone);//CREATE CIRCLE WITH RADIUS DISTANCE OF TWO OBJECTS

		ArrayList<Point2D> points = new ArrayList<Point2D>(Circle2D.circlesIntersections(collisionConeCircle, obstacleCircle));// FIND INTERSECTION POINTS BETWEEN THE TWO CIRCLES
		Vector p1 = new Vector(points.get(0));
		Vector p2 = new Vector(points.get(1));
		
		this.cs = new CircularSector();
		cs.setCenter(this.self.getCenter());//SET CIRCULAR SECTOR CENTER
		cs.setTangentPoints(p1,p2);//SET INTERSECTION POINTS
		cs.setRadius(this.self.getR());//SET SECTOR  RADIUS
		this.trapezoid = new Trapezoid(cs);
		this.trapezoid.createTrapezoid(this.self.getS(),this.obstacle.getS());
		//if(!augmented) this.trapezoid.clip();
		cs.createMinowskiSum(this.obstacle.getS());
	}
	
	public double collisionTime(Vector relativeVelocity){
		Ray2D ray = new Ray2D(this.self.getCenter(),relativeVelocity);
		return this.cs.getPolygon().contains(this.self.getCenter()) ? Double.MIN_VALUE : this.cs.intersection(ray);
	}
	
	public boolean collision(Vector relativeVelocity){
		Ray2D ray = new Ray2D(this.self.getCenter(),relativeVelocity);
		//return this.trapezoid.intersects(ray);
		return this.cs.intersects(ray);
	}
	
	public boolean collision(){
		Ray2D ray = new Ray2D(this.self.getCenter(),this.self.getV());
		//return this.trapezoid.intersects(ray);
		return this.cs.intersects(ray);
	}
	
	public void createVelocityObstacle(Vector VB){
		this.cs.center(VB);
		this.cs.leftPoint(VB);
		this.cs.rightPoint(VB);
		this.trapezoid.updateTrapezoid(VB);
	}
	
	public void createReciprocalVelocityObstacle(Vector VA,Vector VB){
		Vector VAB = new Vector(VA);
		VAB.add(VB);
		VAB.multi(0.5);
		
		this.cs.center(VAB);
		this.cs.leftPoint(VAB);
		this.cs.rightPoint(VAB);
		this.trapezoid.updateTrapezoid(VAB);
	}
	
	public boolean isCollisionFree(boolean augmented){//COLLISION CONE TEST
		Vector relativeVelocity = new Vector(obstacle.getV(),self.getV());
		Vector relativeVelocityPoint = new Vector(this.self.getCenter());
		relativeVelocityPoint.add(relativeVelocity);//FIND POINT AFTER APPLYING THE SPECIFIED VELOCITY
		Ray2D ray = new Ray2D(self.getCenter(),relativeVelocityPoint);
		
		if(augmented) return !trapezoid.containsPoint(relativeVelocityPoint);
		else return !trapezoid.intersects(ray);
		//return !cs.containsPoint(relativeVelocityPoint);
		
	}
	
	public boolean isCollisionFree(Vector va,boolean augmented){//IF VELOCITY ADMISSIBLE RETURN TRUE
		Vector newPoint = new Vector(self.getCenter());
		newPoint.add(va);//FIND NEW LOCATION
		Ray2D ray = new Ray2D(self.getCenter(),newPoint);
		
		if(augmented) return !trapezoid.containsPoint(newPoint);//RETURN FALSE IF VO CONTAINS POINT THUS VELOCITY RESULTS IS COLLISION FREE
		else return !trapezoid.intersects(ray);
		//return !trapezoid.intersects(ray);	
	}
	
	public Thing getObstacle(){
		return this.obstacle;
	}
}
