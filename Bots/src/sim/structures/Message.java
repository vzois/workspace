package sim.structures;

import sim.elements.Bot;

public class Message {
	private Bot bot;
	private Long timestamp;
	private int priority=0;

	public Message(Bot bot,Long timestamp, int priority){
		this.bot = bot;
		this.timestamp = timestamp;
		this.priority = priority;
	}
	
	public Message(Bot bot,Long timestamp){
		this(bot,timestamp,-1);
	}
	
	public Message(Long timestamp){
		this(null,timestamp,-1);
	}
	
	public Message(){
		this(null,new Long(0),-1);
	}
	
	public Bot getBot(){
		return this.bot;
	}
	
	public Long getTimeStamp(){
		return this.timestamp;
	}
	
	public int getPriority(){
		return this.priority;
	}
	
}
