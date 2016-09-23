package sim.elements;
import java.awt.Color;
import java.awt.geom.Line2D;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;

import sim.graphics.Window;
import sim.structures.Message;
import sim.structures.Vector;

public abstract class Bot extends Thing {
	protected double range;
	public Integer id;
	public Random seed;
	protected Long timestamp;
	protected double maxSpeed=1;
	protected boolean flatWorld = false;
	
	protected ConcurrentLinkedQueue<Message> mq;//MESSAGE QUEUE
	private Object[] neighbors;
	
	public Bot(double size,Color color){//BOT WITH SIZE AND COLOR AND INFINITE SIGHT
		this(size,Integer.MAX_VALUE,color);
	}
		
	public Bot(double size,double range, Color color){
		super(size,color);
		this.range=range;
		this.velocity = new Vector(0,0);
		this.mq = new ConcurrentLinkedQueue<Message>();
		this.timestamp= new Long(0);
	}
	
	public void setFlatWorld(boolean flatWorld){
		this.flatWorld = flatWorld;
	}
	
	public double getMaxSpeed(){
		return this.maxSpeed;
	}
	
	public void setMaxSpeed(double maxSpeed){
		this.maxSpeed = maxSpeed;
	}
	
	public double getR(){// GET SENSOR RANGE
		return this.range;
	}
	
	public void setS(int size){
		this.size = size;
	}
	
	public void setV(Vector velocity){
		this.velocity = velocity;
	}
	
	public void move(int LX,int LY){// IMPLEMENT MOVEMENT

		double x = this.getV().getX() + this.getL().getX();
		double y = this.getV().getY() + this.getL().getY();
		if(Window.flatWorld){
			if(x<0 || x>LX-2*this.size || y<0 || y>LY-2*this.size){
				this.velocity.set(-this.getV().getX(),-this.getV().getY());
				x = this.getV().getX() + this.getL().getX();
				y = this.getV().getY() + this.getL().getY();
			}
		}else if(!Window.flatWorld){
			if(x<0) x = LX + x; else if(x>LX) x = LX- x;
			if(y<0) y = LY + y; else if(y>LY) y = LY - y;
		}
		
		this.setL(x,y);
	}
	
	public void stop(){
		this.velocity.set(0, 0);
	}
	
	public void limitSpeed(){
		if(this.velocity.norm()>this.getMaxSpeed()){
			this.velocity.normalize();
			this.velocity.multi(this.getMaxSpeed());
		}
	}
	
	public void setNeighbors(Object[] neighbors){
		this.neighbors = neighbors;
	}
	
	public Object[] getNeighbors(){
		return this.neighbors;
	}
	
	public Message readMessageQueue(){
		return this.mq.poll();
	}
	
	public void writeMessageQueue(Message m){
		this.mq.add(m);
	}
	
	protected void emptyMessageQueue(){
		
	}
	
	public void sleep(int secs){
		try{
			Thread.sleep(secs);
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public Line2D.Double getVVShape(){
		Vector start = new Vector(this.getCenter());
		Vector end = Vector.add(this.getCenter(), this.getV());
		Line2D.Double l2D = new Line2D.Double(start.x(),start.y(),end.x(),end.y());
		return l2D;
	}
	
	public abstract void act();
		
}
