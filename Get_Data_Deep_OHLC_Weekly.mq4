//+------------------------------------------------------------------+
//|                                                      File_01.mq4 |
//|                        Copyright 2014, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+

#property copyright "Copyright 2014, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
#property strict


input string inputFile = "GBPUSD_WeeklyData.csv";
input string symbol = "GBPUSD";
input ENUM_TIMEFRAMES timeframe = PERIOD_W1;
input int min_len_confirm_bar_direction = 5;
input int bar_count = 7000; 
//-- for daily 7200 ; H4:35000

double open,close,high,low,vol,open_next,close_next;
double bodyLenght,candleLenght,std_dev;
datetime time;
double rsi;
string time_str;
datetime curDate;

//-- down=-1 ; up=1 ; non_direction=0 for the next candle that is coming
int direction=0; 
int sar_status=0;
int stock_status=0;
int adx_status=0;

double adx_main=0;
double adx_min =0;
double adx_max =0;
double stoch=0;
double ema_3=0;
double ema_10=0;
double ema_30=0;
double ema_50=0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(){
   int filehandle=FileOpen(inputFile, FILE_WRITE|FILE_CSV,',');
   if(filehandle==-1)                      
   {
      Alert("An error while opening the file. ");
      PlaySound("Bzrrr.wav");         
      return(-1);                          
   }
   //-----------------------------------------------------------------//
   FileWrite(filehandle, "Date", "Open","High","Low","Close","Volume") ;
   
   for(int index=bar_count; index>=0; index--)
   {
      get_data(index);
      FileWrite(filehandle,curDate, open, high, low, close, vol); 
   }
   
   FileClose(filehandle);
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
 
      
}
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+
void get_data(int positon){
   
   curDate = iTime(symbol, timeframe, positon); 
   time_str = StringFormat("%02i-%02i-%i" ,TimeDay(curDate), TimeMonth(curDate), TimeYear(curDate) );
   
   open = iOpen(symbol, timeframe, positon);
   high = iHigh(symbol, timeframe, positon);
   low = iLow(symbol, timeframe, positon);
   close = iClose(symbol, timeframe, positon);
   vol = (double)iVolume(symbol, timeframe, positon);
   rsi = iRSI(symbol,timeframe,14,PRICE_CLOSE,positon);
   adx_main = iADX(symbol, timeframe, 14, PRICE_CLOSE, MODE_MAIN, positon);
   adx_max = iADX(symbol, timeframe, 14, PRICE_CLOSE, MODE_PLUSDI, positon);
   adx_min = iADX(symbol, timeframe, 14, PRICE_CLOSE, MODE_MINUSDI, positon);  
   stoch = iStochastic(symbol,timeframe,5,3,3,MODE_EMA,STO_LOWHIGH,MODE_MAIN,positon);
   ema_3 = iMA(symbol, timeframe, 3, 0, MODE_EMA, PRICE_CLOSE, positon);
   ema_10 = iMA(symbol, timeframe, 10, 0, MODE_EMA, PRICE_CLOSE, positon);
   ema_30 = iMA(symbol, timeframe, 30, 0, MODE_EMA, PRICE_CLOSE, positon);
   ema_50 = iMA(symbol, timeframe, 50, 0, MODE_EMA, PRICE_CLOSE, positon);
   std_dev = iStdDev(symbol, timeframe, 5, 0, MODE_SMA, PRICE_CLOSE, positon);
   bodyLenght = close - open;
   candleLenght = high - low;
   
   //----------------------------------------------------------+
   sar_status = SAR_OK(0.02,0.2, positon);
   adx_status = ADX_OK(14, 20, positon);
   stock_status = Stochastic_OK(5,3,3, positon);
   //----------------------------------------------------------+
   //if(positon==0)
   //   return;
   open_next = iOpen(symbol, timeframe, positon-1);
   close_next = iClose(symbol, timeframe, positon-1);
   if (open_next>close_next)  //&& // ((open_next - close_next)>min_len_confirm_bar_direction*Point) )
      direction = -1;
   //else if(open_next<close_next) // && ((close_next - open_next) > min_len_confirm_bar_direction*Point) )
     // direction = 1;
   else
      direction = 1;
}  
//+------------------------------------------------------------------+
//-- step=0.02,max=0.2
int SAR_OK(double Step,double Max, int pos)
{
   double SAR_Upper_Time_0=iSAR(symbol,timeframe+1,Step,Max,pos);
   double SAR0=iSAR(symbol,timeframe,Step,Max,pos);
   if(SAR0>iClose(symbol, timeframe, pos))
      return(-1);
   else if(SAR0<iClose(symbol, timeframe, pos))
      return(1);
   else return(0);
}

//+------------------------------------------------------------------+
int ADX_OK (int AdxPeriod,int ADX_Min, int pos) // count is the number of tick that we chek in ADX
{                      // if retunn 1 means that adx + ; return -1 means that adx - ; return 0 means that no comment ; 
   
   double CurADX_0_Main = iADX(symbol, timeframe, AdxPeriod, PRICE_CLOSE, MODE_MAIN, pos);
   double CurADX_0_DI_Plus = iADX(symbol, timeframe, AdxPeriod, PRICE_CLOSE, MODE_PLUSDI, pos);
   double CurADX_0_DI_Minus = iADX(symbol, timeframe, AdxPeriod, PRICE_CLOSE, MODE_MINUSDI, pos);      
   
   // چک برای صعودی بودن شرایط
   bool Up_Down_Condition = (CurADX_0_Main > ADX_Min) ;
   bool Up_Condition = (CurADX_0_DI_Plus > CurADX_0_DI_Minus) ;
   bool Down_Condition = (CurADX_0_DI_Plus < CurADX_0_DI_Minus) ;
   
   if (Up_Condition && Up_Down_Condition)
      return (1) ;
   else if (Down_Condition && Up_Down_Condition)
      return (-1) ;
   else 
      return (0) ;
}
//+------------------------------------------------------------------+
//-- 5,3,3 input 
int Stochastic_OK(int Stoch_KPeriod,int Stoch_DPeriod,int Stoch_Slow,int pos)
{
   double Stoch_M0=iStochastic(symbol,timeframe,Stoch_KPeriod,Stoch_DPeriod,Stoch_Slow,MODE_EMA,STO_LOWHIGH,MODE_MAIN,pos);
   double Stoch_M1=iStochastic(symbol,timeframe,Stoch_KPeriod,Stoch_DPeriod,Stoch_Slow,MODE_EMA,STO_LOWHIGH,MODE_MAIN,pos+1);
   if(Stoch_M0<30)    //&&Stoch_M1<=20)
      return(1);
   else if(Stoch_M0>70)   //&&Stoch_M1>=80)
      return(-1);
   else
      return(0);
}