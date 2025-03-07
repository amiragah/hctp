//+------------------------------------------------------------------+
//|In this version, we recognise candle stick pattern including :    |
//|Engulfing Bullish, Star, Grave stone, Dragon fly, Inverted Hammer |
//|After that we open them by watching weekly trend deep prediction  |                 |
//+------------------------------------------------------------------+
//
//
//

#property copyright "Copyright 2023, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00" 
#property strict

input ENUM_TIMEFRAMES TLB_TimeFrame=PERIOD_W1;
input ENUM_TIMEFRAMES ExpertDefualt=PERIOD_D1;
input double Lot=1;         

input int MaxTradesNumer=5;
input string inputSymbol = "GBPUSD";
double SL,TP;
string subject_not, message_not;
int Slippage = 5;

string predictionfilename = "d:\\AmirAgah\\Model_Saved\\mql_csv_data02.csv";
// Structure to hold the data from the CSV
struct Prediction {
   datetime from_date;
   datetime to_date;
   bool prediction;
};

// Global array to store the predictions
Prediction predictions[], ontick_predictions[];
bool onTickPredictioArrayFlag = true;
int predictionOpenFileCounter=0;

// Global string variable to buffer trade information
string tradeDataBuffer = "";

//--------------------------------------------------------//Pattern 01   
extern bool      FindInvertedHammer=true;
extern int       MinLengthOfUpTail_01=48;
extern int       MaxLengthOfLoTail_01=22;
extern double    MaxLengthOfBody_01=36; 
extern double    MinLengthOfBody_01=14; 
extern double    MinLengthOfBodyConfirmInvertedHammer=30;
extern bool      SLTP_input_InvertedHammer=false;
extern double    TP_InvertedHammer=140;
extern double    SL_InvertedHammer=110;

//--------------------------------------------------------//Pattern 02   
extern bool      FindEngulfingBullish=true;     
extern double    MinLengthOfBody=32;
//-- extern double    MinLengthOfMainBody=32;
//-- Dont Use
extern double    MaxLengthOfBody=524;
extern int       TPPercent_02=96;
extern bool      SLTP_input_EngulfingBullish=false;
extern double    TP_EngulfingBullish=100;
extern double    SL_EngulfingBullish=100;

//-------------------------------------------------------//Pattern 04   
extern bool      FindDragonFlyDojiBullish=true;   
extern int       MaxLengthOfUpTail_04=16; 
//candle with upper tail equal or more than this will show up
extern int       MinLengthOfLoTail_04=42; 
//candle with lower tail equal or more than this will show up
extern double    MaxLengthOfBody_04=12; 
//candle with body less or equal with this will show up
extern double    MinLengthOfBody_04=0;
extern bool      SLTP_Input_DragonFly=false;
extern double    TP_DragonFly=100;
extern double    SL_DragonFly=80;

//-------------------------------------------------------//Pattern 06 
extern bool      FindStars=true;   
//--  For 3'th candle (Last candle):
extern int       MinLengthOfBodyForLastCandleSell_06=68;
extern int       MinLengthOfBodyForLastCandleBuy_06=34;
       int       MaxLengthOfBodyForLastCandle_06=96;
//--  For 2'th candle (middle candle):
extern int       MaxLenghtOfBodyForSecondCandleSell_06 = 10;
extern int       MaxLenghtOfBodyForSecondCandleBuy_06 = 14;
       int       BodyToFullCandlePercent_06=50;
//--  For 1'th candle (Last candle):
extern int       MinLengthOfBodyForFirstCandleSell_06=66;
extern int       MinLengthOfBodyForFirstCandleBuy_06=66;
extern int       PercentStar=60;
extern int       TPPercent_Start=110;
int SLTP_MorePipFromShadow_Star = 4;
//-------------------------------------------------------//Pattern 08 - Grave Stone  
extern bool      FindGraveStone=true;  
extern int       MinLengthOfUpTail_08=75; 
//candle with upper tail equal or more than this will show up
extern int       MaxLengthOfLoTail_08=27; 
//candle with lower tail equal or more than this will show up
extern double    MaxLengthOfBody_08=21; 
//candle with body less or equal with this will show up
extern double    MinLengthOfBody_08=1;
extern bool      SLTP_Input_GraveStone=false;
extern double    TP_GraveStone=120;
extern double    SL_GraveStone=120;
//--------------------------------------------------------//Pattern 09   ___DONE___
extern bool      FindEngulfingBearish=false;     
//-- We use max lenght of body for both side Long and Short , then this parameter ar note use now
extern double    MinLengthOfBodyBearish_9=38;
extern double    MaxLengthOfBodyBearish_9=832;
extern int       TPPercent_09=200;
extern bool      SLTP_input_EngulfingBearish=false;
extern double    TP_EngulfingBearish=100;
extern double    SL_EngulfingBearish=100;


input bool   MyBrokerHas5Digits=false; 
input double SARStep=0.029;
input double SARMax=0.41;

double pt=0, pt1=0;

bool IsNewBar=false;
datetime MyTime;
int spread = 3;
int SLTP_MorePipFromShadow=1;


int OnInit()
  {
   
   subject_not = "Test Subject";
   message_not = "Test Message";
   
   Send_Notification_All(subject_not, message_not);
  
   MyTime=Time[0];
   if(Digits<4) pt1=0.01;
   else pt1=0.0001;
   if(MyBrokerHas5Digits){
      if(Digits<4) pt=0.001;
      else pt=0.00001;
   }
   else{
      pt=pt1;
   }
     
   /*if(ArraySize(predictions) == 0) { // Read the CSV only once at the start.
       if(ReadPredictionsFromCSV(predictionfilename) != 0) { // Replace "predictions.csv" with your file name.
           Print("Failed to read predictions. Exiting.");
           //return; // Or handle the error as needed.
       }
   }     
     
   //-- CHECK HERE NOT NEcessary
   int res = PredictionTrendStatus("2017.01.09");
   */
     
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // *** Write the buffer to the file at the end of the testing process
      if (tradeDataBuffer != "") { // Only write if there's data in the buffer
      if (!WriteTradeInfoToCSV("trade_history.csv")) {
         Print("Error writing to CSV.");
      }
   }

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(MyTime<Time[0])
   {
      IsNewBar=true;
      MyTime=Time[0];
   }
   else
      return;
       
   //---------------------------------    Creating the dictionary for trend prediction Area -------------------------------//  
   
   if(onTickPredictioArrayFlag) { // Read the CSV only once at the start.
       if(ReadPredictionsFromCSV("mql_csv_data02.csv") != 0) { // Replace "predictions.csv" with your file name.
           Print("Failed to read predictions. Exiting.");
           return; // Or handle the error as needed.
       }
       predictionOpenFileCounter = predictionOpenFileCounter + 1;
   }     
   onTickPredictioArrayFlag = false;   //-- To have it just one!
     
   
   //int res = PredictionTrendStatus(TimeCurrent());   
   //Alert("Our new prediction size is ::::::::::::::::::::::::::::::: ",ArraySize(predictions),"DATE ::: ",TimeCurrent(),"  RESult::: ",res);
   
   
   //--------------------------------------------End Trend prediction Area ------------------------------------------------//    
       
   //---------------------------------------------------------- Condition section -----------------------------------------//
   //-- Inverted Hammer section
   int ConditionBodyConfirmInvertedHammer=BodyConfirmInvertedHammer(inputSymbol,2,MinLengthOfBodyConfirmInvertedHammer);
   int ConditionInvertedHammer=FindInvertedHammerMethod(inputSymbol, ExpertDefualt, 1);    
   //---------------------------------------------------------------------------------------
   //-- Engulfing section
   int ConditionEngulf_Sell_01=FindBearishEngulfingMethod(inputSymbol, ExpertDefualt, 1, 1);
   int ConditionEngulf_Buy_01=FindBullishEngulfingMethod(inputSymbol, ExpertDefualt, 1, 1);
   int ConditionEngulf_03 = BodyConfirm(inputSymbol, 1, MaxLengthOfBody, ExpertDefualt);
   int ConditionEngulf_04 = BodyMoreThanX(inputSymbol, 2, MinLengthOfBody, ExpertDefualt);  
   //-- Engulfing check with Bearish parameters
   int ConditionEngulf_13 = BodyConfirm(inputSymbol, 1, MaxLengthOfBodyBearish_9, ExpertDefualt);
   int ConditionEngulf_14 = BodyMoreThanX(inputSymbol, 2, MinLengthOfBodyBearish_9, ExpertDefualt); 
   //-- DragonFly section
   int ConditionDragonFly=FindDragonFlyDojiMethod(inputSymbol,ExpertDefualt,1);
   //-- Stars section
   int Cond_EveningStar = FindEveningStarMethod(inputSymbol, ExpertDefualt, 1, 0);
   int Cond_MorningStar = FindMorningStarMethod(inputSymbol, ExpertDefualt, 1, 0); 
   //-- Grave Stone Section
   int ConditionGraveStone=FindGraveStoneDojiMethod(inputSymbol,ExpertDefualt,1);
   
   
//------------------------------------------------------ MAIN Section for TRADING -----------------------------------------//

   
   if(OrdersTotal()<=MaxTradesNumer && IsNewBar )   //-- شرط ورود به معامله داریم
   {    
   
      if(FindGraveStone && ConditionGraveStone && PredictionTrendStatus(TimeCurrent())==1)
      {

         //if(SLTP_Input_GraveStone==false)         
         if(SLTP_Input_GraveStone==false)
         {
            if(OrderSend(inputSymbol,OP_SELL,Lot,Bid,3,iHigh(inputSymbol, ExpertDefualt,1), Bid-((iHigh(inputSymbol, ExpertDefualt,1)-Bid)),"Candle_GBPUSD_D1",34,0,clrRed)<0)
               return;
            else
               {
                  //int res = PredictionTrendStatus(TimeCurrent());
               
                  // Write trade info to Trade Buffer
                  AddTradeInfoToBuffer(TimeCurrent(), Ask, "SELL", "GraveStone", Lot, 0, 0);  
                     
                  IsNewBar=false;
                  return;
               }
         }
         //else if(SLTP_Input_GraveStone)
         else if(SLTP_Input_GraveStone)
         {
            if(OrderSend(inputSymbol,OP_SELL,Lot,Bid,3, Bid + SL_GraveStone*Point, Bid - TP_GraveStone*Point,"Candle_GBPUSD_D1",34, 0, clrRed)<0)
               return;
            else
               {
                  //int res = PredictionTrendStatus(TimeCurrent());
               
                  // Write trade info to Trade Buffer
                  AddTradeInfoToBuffer(TimeCurrent(), Ask, "SELL", "GraveStone", Lot, 0, 0);                      
                     
                  IsNewBar=false;
                  return;
               }
         }
      }
      
      //---------------------------------- Inverted Hammer Section  -----------------------//
      //if(FindInvertedHammer && ConditionInvertedHammer==1 && ConditionBodyConfirmInvertedHammer==1)
      if(FindInvertedHammer && ConditionInvertedHammer==1 && ConditionBodyConfirmInvertedHammer==1 && PredictionTrendStatus(TimeCurrent())==0)
      {
         if(SLTP_input_InvertedHammer==false)
         {
            if(OrderSend(Symbol(),OP_BUY, Lot*3 ,Ask,Slippage,Low[2],Ask+((Ask-Low[2])),"Candle_GBPUSD_D1",34,0,clrGreen)<0)
               return;
            else
               {
                  //int res = PredictionTrendStatus(TimeCurrent());
               
                  // Write trade info to Trade Buffer
                  AddTradeInfoToBuffer(TimeCurrent(), Ask, "BUY", "InvertedHammer", Lot, 0, 0);                      
                     
                  IsNewBar=false;
                  return;
               }
         }
         else if(SLTP_input_InvertedHammer)
         {
            if(OrderSend(Symbol(),OP_BUY, Lot*3 ,Ask,Slippage,Ask-SL_InvertedHammer*Point,Ask+TP_InvertedHammer*Point,"Candle_GBPUSD_D1",34,0,clrGreen)<0)
               return;
            else
               {
                  //int res = PredictionTrendStatus(TimeCurrent());
                  
                  // Write trade info to Trade Buffer
                  AddTradeInfoToBuffer(TimeCurrent(), Ask, "BUY", "InvertedHammer", Lot, 0, 0);                      

                  IsNewBar=false;
                  return;
               }
         }
      }
   
   
      //---------------------------------- Engulfing Bearish Section  -----------------------//
      //if(FindEngulfingBearish)    //-- Engulfing Bearish  ok .
      if(FindEngulfingBearish && PredictionTrendStatus(TimeCurrent())==1)    //-- Engulfing Bearish  ok .
      {
         //-- Short positions in bearish signals
         if(ConditionEngulf_Sell_01==-1 && ConditionEngulf_13==-1 && ConditionEngulf_14==-1)
         {
            Send_Notification_All("Order","Engulfing is recognized!");
            
            if(CalculateSLTP_Engulfing(inputSymbol,OP_SELL,1,TPPercent_09)==false)
               return;
            if(OrderSend(inputSymbol,OP_SELL,Lot,Bid,Slippage,SL,TP,"Candle_GBPUSD_D1",34,0,clrRed)<0)  
               return;
            else
               {
                  //int res = PredictionTrendStatus(TimeCurrent());
                  
                  // Write trade info to Trade Buffer
                  AddTradeInfoToBuffer(TimeCurrent(), Ask, "SELL", "Engulfing Bearish", Lot, 0, 0);                      

                  IsNewBar=false;   
                  return;
               }
         }
      }
      
      //---------------------------------- Engulfing Bullish Section  -----------------------//
      //if(FindEngulfingBullish)    //-- Engulfing  ok .
      if(FindEngulfingBullish && PredictionTrendStatus(TimeCurrent())==0)    //-- Engulfing  ok .
      {             
         //-- Long position in Bullish signals
         if(ConditionEngulf_Buy_01==1 && ConditionEngulf_04==1 && ConditionEngulf_03==1 )
         {
            Send_Notification_All("Order","Engulfing is recognized!");
            
            if(CalculateSLTP_Engulfing(inputSymbol,OP_BUY,2,TPPercent_02)==false)
               return;
            if(OrderSend(Symbol(),OP_BUY,Lot,Ask,Slippage,SL,TP,"Candle_GBPUSD_D1",34,0,clrGreen)<0)
               return;
            else
               {
                  //int res = PredictionTrendStatus(TimeCurrent());
                  
                  // Write trade info to Trade Buffer
                  AddTradeInfoToBuffer(TimeCurrent(), Ask, "BUY", "Engulfing Bullish", Lot, 0, 0);                                         
                     
                  IsNewBar=false;   
                  return;
               }
         }
 
      }          //-- End of Engulfing bullish ok .  
      
   
      //---------------------------------- DragonFly Section  -----------------------//
      //if(FindDragonFlyDojiBullish && ConditionDragonFly==1)    //-- DragonFly  ok .
      if(FindDragonFlyDojiBullish && ConditionDragonFly==1 && PredictionTrendStatus(TimeCurrent())==0)    //-- DragonFly  ok .
      {   
         Send_Notification_All("Order","Dragon fly is recognized!");
         
         if(SLTP_Input_DragonFly==false)
         {
            //-- here I multiply LOT by 2
            if(OrderSend(inputSymbol,OP_BUY,Lot*2,Ask,Slippage,iLow(inputSymbol, ExpertDefualt, 1),Ask+((Ask-iLow(inputSymbol, ExpertDefualt, 1))),"Candle_GBPUSD_D1",34,0,clrGreen)<0)
               return;
            else
            {
               //int res = PredictionTrendStatus(TimeCurrent());
               
               // Write trade info to Trade Buffer
               AddTradeInfoToBuffer(TimeCurrent(), Ask, "BUY", "Dragon Fly", Lot, 0, 0);                                         
                     
                     
               IsNewBar=false;
               return;
            }
         }
         else if(SLTP_Input_DragonFly)
         {
            if(OrderSend(inputSymbol,OP_BUY,Lot,Ask,Slippage,Ask-SL*Point,Ask+TP*Point,"Candle_GBPUSD_D1",34,0,clrGreen)<0)
               return;
            else
            {
               //int res = PredictionTrendStatus(TimeCurrent());
               //if (!WriteTradeInfoToCSV("trade_history.csv", TimeCurrent(), Ask, "BUY", "Dragon Fly", Lot, 0, 0) )
                 //    Print("Error writing trade info to CSV.");
               // Write trade info to Trade Buffer
               AddTradeInfoToBuffer(TimeCurrent(), Ask, "BUY", "Dragon Fly", Lot, 0, 0);                                         
               
               IsNewBar=false;
               return;
            }    
         }    
      
      }          //-- End of Dragonfly bullish ok . 
      
      
      
       
      //---------------------------------- Stars Section  -----------------------//
      //---------------------------------- Evening star
      //if(FindStars && Cond_EveningStar==-1)
      if(FindStars && Cond_EveningStar==-1 && PredictionTrendStatus(TimeCurrent())==1)
      {
         Send_Notification_All("Order","Stars is recognized!");
         
         if(CalculateSLTP_Stars(inputSymbol,OP_SELL,1,TPPercent_Start)==false)
            return;
         //-- here I multiply LOT by 2
         if(OrderSend(inputSymbol,OP_SELL,Lot*2,Bid,Slippage,SL,TP,"Candle_GBPUSD_D1",34,0,clrRed)<0)  
            return;
         else
            {
               //int res = PredictionTrendStatus(TimeCurrent());
               // Write trade info to Trade Buffer
               AddTradeInfoToBuffer(TimeCurrent(), Ask, "SELL", "EVENING STAR", Lot, 0, 0);                                         
                     
               IsNewBar=false;   
               return;
            }
      }
      //---------------------------------- Morning star
      //if(FindStars && Cond_MorningStar==1 )
      if(FindStars && Cond_MorningStar==1 && PredictionTrendStatus(TimeCurrent())==0)
      {
         Send_Notification_All("Order","Stars is recognized!");
         
         if(CalculateSLTP_Stars(inputSymbol,OP_BUY,2,TPPercent_Start)==false)
            return;
         //-- here I multiply LOT by 2
         if(OrderSend(inputSymbol,OP_BUY,Lot*2,Ask,Slippage,SL,TP,"Candle_GBPUSD_D1",34,0,clrGreen)<0)
            return;
         else
            {    
               //int res = PredictionTrendStatus(TimeCurrent());               
               // Write trade info to Trade Buffer
               AddTradeInfoToBuffer(TimeCurrent(), Ask, "BUY", "Morning Star", Lot, 0, 0);                                         

               IsNewBar=false;   
               return;
            }
      }   //-- End of Stars . 
      
      
      
      
   }  // -- end of entering condition loop  //-----------------------------------


}


//--------------------------------------------------------------------------------------------------------//

// Function to read the CSV file and populate the predictions array
int ReadPredictionsFromCSV(string filename) 
{
   // -- Reset prediction array here.
   ArrayResize(predictions, 0);

   int file_handle = FileOpen("mql_csv_data02.csv", FILE_READ | FILE_TXT | FILE_CSV);
   //int file_handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_CSV);
   if(file_handle == INVALID_HANDLE) 
   {
      Print("Error opening file: ", filename);
      return -1; // Indicate error
   }

   // Read header line (if any) and discard it
   string header_line = FileReadString(file_handle); // Assuming there's a header

   while(!FileIsEnding(file_handle)) 
   {
      string line = string(FileReadString(file_handle));
      //string newline = string(line);
      // Pre-allocate the parts array.  A size of 10 should be enough for most cases.
      string parts[10];  // Fixed size array    
      int count = StringSplit(line,',',parts); // Split by comma

      if(count >= 3) { // Ensure we have 3 parts
     
         string fromdateStr = string(parts[1]);
         int formattedDateStr = StringReplace(fromdateStr, "-", "."); // Replace all "-" with "."
         datetime from_date = StrToTime(fromdateStr);
        
         string todateStr = string(parts[2]);
         int formattedDateStr2 = StringReplace(todateStr,"-", "."); // Replace all "-" with "."
         datetime to_date = StrToTime(todateStr);
         
         bool prediction = StringToInteger(parts[3]) == 1; // 1 for true, 0 for false   ////???????????????????????????????????????????????
      

         Prediction p;
         p.from_date = from_date;
         p.to_date = to_date;
         p.prediction = prediction;

         ArrayResize(predictions, ArraySize(predictions) + 1);
         predictions[ArraySize(predictions) - 1] = p;
      }
   }

   FileClose(file_handle);
   return 0; // Success
}
//--------------------------------------------------------------------------------------------------------//
//-------------------------- Function to check the prediction for the current time -----------------------//
int PredictionTrendStatus(datetime current_time) {
   int len = ArraySize(predictions);
   for(int i = 0; i < ArraySize(predictions); i++) 
   {
      if(current_time >= predictions[i].from_date && current_time <= predictions[i].to_date) {
         return predictions[i].prediction;
      }
   }
   return -100; // 0=false / -1=Bearish / +1=Bullish
}
//--------------------------------------------------------------------------------------------------------//
//------------------------------------ Sending notifications section -------------------------------------//

void Send_Notification_All(string subject, string message)
{
   SendMail(subject, message);
   SendNotification(message);
}

//---------------------------------------------
int SAR_OK(string symbol,double Step,double Max)
{
   double SAR0=iSAR(symbol,0,Step,Max,0);
   if(SAR0>Close[0])
      return(-1);
   else if(SAR0<Close[0])
      return(1);
   else return(0);
}
//---------------------------------------------------------------------------



//+---------------------------------------------------------------+//
//-----------------------------------------------------------------//
//--                       Grave Stone method                    --// 
//-----------------------------------------------------------------//
int FindGraveStoneDojiMethod(string symbol, int timeFrame,  int candleNo)
{  
   double C=iClose(symbol,timeFrame,candleNo);
   double O=iOpen(symbol,timeFrame,candleNo);
   double H=iHigh(symbol,timeFrame,candleNo);
   double L=iLow(symbol,timeFrame,candleNo);
      
   if(   (C>O && (H-C)>=MinLengthOfUpTail_08*pt && (O-L)<=MaxLengthOfLoTail_08*pt && MathAbs(C-O)<=MaxLengthOfBody_08*pt && MathAbs(C-O)>=MinLengthOfBody_08*pt) 
      || (C<O && (H-O)>=MinLengthOfUpTail_08*pt && (C-L)<=MaxLengthOfLoTail_08*pt && MathAbs(C-O)<=MaxLengthOfBody_08*pt && MathAbs(C-O)>=MinLengthOfBody_08*pt))
   {
      return(1);
   }
   else
   {
      return(0);
   }
   
}
//-----------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//-----------------------------------------------------------------//
//--                    inverted hammer method                   --// 
//-----------------------------------------------------------------//
int FindInvertedHammerMethod(string symbol, int timeFrame,  int candleNo)
{  
   double C=iClose(symbol,timeFrame,candleNo);
   double O=iOpen(symbol,timeFrame,candleNo);
   double H=iHigh(symbol,timeFrame,candleNo);
   double L=iLow(symbol,timeFrame,candleNo);
      
   if((C>O && H-C>=MinLengthOfUpTail_01*pt && O-L<=MaxLengthOfLoTail_01*pt  && MathAbs(C-O)<=MaxLengthOfBody_01*pt && MathAbs(C-O)>=MinLengthOfBody_01*pt)
              || (O>C && H-O>=MinLengthOfUpTail_01*pt && C-L<=MaxLengthOfLoTail_01*pt  && MathAbs(C-O)<=MaxLengthOfBody_01*pt && MathAbs(C-O)>=MinLengthOfBody_01*pt))
   {
      //Comment("fing Inverted Hammer !");
      return(1);
   }
   else
   {
      return(0);
   }
   
}
//-----------------------------------------------------------------------------------
int BodyConfirmInvertedHammer(string symbol, int candleNo, double MaxLenght)
{
   double C_Confirm=iClose(symbol,ExpertDefualt,candleNo);
   double O_Confirm=iOpen(symbol,ExpertDefualt,candleNo);
   double H_Confirm=iHigh(symbol,ExpertDefualt,candleNo);
   double L_Confirm=iLow(symbol,ExpertDefualt,candleNo);
   
   if((O_Confirm<C_Confirm)&&(C_Confirm-O_Confirm)>=MaxLenght*Point)
      return(-1);
   else if((C_Confirm<O_Confirm)&&(O_Confirm-C_Confirm)>=MaxLenght*Point)
      return(+1);
   else
      return(0);
}
//------------------------------------------------------------------------------------
//-----------------------------------------------------------------//
//--                              stars methods                  --// 
//-----------------------------------------------------------------//
int FindEveningStarMethod(string symbol, int timeFrame,  int candleNo, int starType)
{  
//-- startype declare the type of second candle in model that may be doji or ect.Default is 0 
//-- which means that it recognize all kind of model.
//-- if statuse is ready for buy it return "1".     

   double C=iClose(symbol,timeFrame,candleNo);
   double O=iOpen(symbol,timeFrame,candleNo);
   double H=iHigh(symbol,timeFrame,candleNo);
   double L=iLow(symbol,timeFrame,candleNo);
   
   double C_1=iClose(symbol,timeFrame,candleNo+1);
   double O_1=iOpen(symbol,timeFrame,candleNo+1);
   double H_1=iHigh(symbol,timeFrame,candleNo+1);
   double L_1=iLow(symbol,timeFrame,candleNo+1);
   
   double C_2=iClose(symbol,timeFrame,candleNo+2);
   double O_2=iOpen(symbol,timeFrame,candleNo+2);
   double H_2=iHigh(symbol,timeFrame,candleNo+2);
   double L_2=iLow(symbol,timeFrame,candleNo+2);
     
   if( ((C_2-O_2)>MinLengthOfBodyForFirstCandleSell_06*pt ) && ( MathAbs(C_1 - O_1)<MaxLenghtOfBodyForSecondCandleSell_06*pt) && ((O-C)>MinLengthOfBodyForLastCandleSell_06*pt)  )
   {
      //Comment("fing Evening Star");
      return(-1);
   }
   else
   {
      return(0);
   }
}

//+------------------------------------------------------------------+
int FindMorningStarMethod(string symbol, int timeFrame,  int candleNo, int starType)
{  
//-- startype declare the type of second candle in model that may be doji or ect.Default is 0 
//-- which means that it recognize all kind of model.
//-- if statuse is ready for buy it return "-1".
   
   double C=iClose(symbol,timeFrame,candleNo);
   double O=iOpen(symbol,timeFrame,candleNo);
   double H=iHigh(symbol,timeFrame,candleNo);
   double L=iLow(symbol,timeFrame,candleNo);
   
   double C_1=iClose(symbol,timeFrame,candleNo+1);
   double O_1=iOpen(symbol,timeFrame,candleNo+1);
   double H_1=iHigh(symbol,timeFrame,candleNo+1);
   double L_1=iLow(symbol,timeFrame,candleNo+1);
   
   double C_2=iClose(symbol,timeFrame,candleNo+2);
   double O_2=iOpen(symbol,timeFrame,candleNo+2);
   double H_2=iHigh(symbol,timeFrame,candleNo+2);
   double L_2=iLow(symbol,timeFrame,candleNo+2);
   
   double C_3=iClose(symbol,timeFrame,candleNo+3);
   double O_3=iOpen(symbol,timeFrame,candleNo+3);
   double H_3=iHigh(symbol,timeFrame,candleNo+3);
   double L_3=iLow(symbol,timeFrame,candleNo+3);
      
   if( ((O_2-C_2)>MinLengthOfBodyForFirstCandleBuy_06*pt ) && ( MathAbs(C_1 - O_1)<MaxLenghtOfBodyForSecondCandleBuy_06*pt) && ((C-O)>MinLengthOfBodyForLastCandleBuy_06*pt)  )
   {
      return(1);
   }
   else
   {
      return(0);
   }   
}

bool CalculateSLTP_Stars(string symbol, int orderType, int patternNo, int tpPercent)
{
   //-- SLTP_MorePipFromShadow is an input parameter that added in calculating SL & TP to the pattern SL. 
   //-- Pattern No is : "1 :BearishEngulffing" - "2 :BullishEngulfing" -  
   double cur_SL,cur_TP;
   switch (patternNo)
   {
         
      
      case 1:   //-- Bearish Engulfing.
      {
      
         if(iHigh(symbol, ExpertDefualt, 1)>iHigh(symbol, ExpertDefualt, 2))
         {cur_SL=iHigh(symbol, ExpertDefualt, 1)+SLTP_MorePipFromShadow_Star*Point;cur_TP = Bid - (iHigh(symbol, ExpertDefualt, 1)-Ask)*tpPercent/100;}
         else 
         {cur_SL=iHigh(symbol, ExpertDefualt, 2)+SLTP_MorePipFromShadow_Star*Point;cur_TP = Bid - (iHigh(symbol, ExpertDefualt, 2)-Ask)*tpPercent/100;}; 
         
         SL = cur_SL;
         TP = cur_TP;
         return(true);                 
      }
      case 2:   //-- Bullish Engulfing.
      {
      
         if(iLow(symbol, ExpertDefualt, 1)<iLow(symbol, ExpertDefualt, 2))
         {cur_SL=iLow(symbol, ExpertDefualt, 1)-SLTP_MorePipFromShadow_Star*Point;cur_TP = Ask + (Ask-iLow(symbol, ExpertDefualt, 1))*tpPercent/100;}
         else 
         {cur_SL=iLow(symbol, ExpertDefualt, 2)-SLTP_MorePipFromShadow_Star*Point;cur_TP = Ask + (Ask-iLow(symbol, ExpertDefualt, 2))*tpPercent/100;};               

         SL = cur_SL;
         TP = cur_TP;
         return(true);                 
      }
      default:
         return(false);
   }
}
//-----------------------------------------------------------------//
//--                          End stars methods                  --// 
//-----------------------------------------------------------------//
//----------------------------------------------------------------------------

//-----------------------------------------------------------------//
//--                          DragonFly methods                  --// 
//-----------------------------------------------------------------//
int FindDragonFlyDojiMethod(string symbol, int timeFrame,  int candleNo)
{  
   double C=iClose(symbol,timeFrame,candleNo);
   double O=iOpen(symbol,timeFrame,candleNo);
   double H=iHigh(symbol,timeFrame,candleNo);
   double L=iLow(symbol,timeFrame,candleNo);
   
   double C_1=iClose(symbol,timeFrame,candleNo+1);
   double O_1=iOpen(symbol,timeFrame,candleNo+1);
   double H_1=iHigh(symbol,timeFrame,candleNo+1);
   double L_1=iLow(symbol,timeFrame,candleNo+1);
   double C_2=iClose(symbol,timeFrame,candleNo+2);
   double O_2=iOpen(symbol,timeFrame,candleNo+2);
   double H_2=iHigh(symbol,timeFrame,candleNo+2);
   double L_2=iLow(symbol,timeFrame,candleNo+2);
   double C_3=iClose(symbol,timeFrame,candleNo+3);
   double O_3=iOpen(symbol,timeFrame,candleNo+3);
   double H_3=iHigh(symbol,timeFrame,candleNo+3);
   double L_3=iLow(symbol,timeFrame,candleNo+3);
      
   if(  (C>O && (H-C)<=MaxLengthOfUpTail_04*pt && (O-L)>=MinLengthOfLoTail_04*pt  && MathAbs(C-O)<=MaxLengthOfBody_04*pt && MathAbs(C-O)>=MinLengthOfBody_04*pt) 
     || (C<O && (H-O)<=MaxLengthOfUpTail_04*pt && (C-L)>=MinLengthOfLoTail_04*pt  && MathAbs(C-O)<=MaxLengthOfBody_04*pt && MathAbs(C-O)>=MinLengthOfBody_04*pt))
   {
      //Comment("fing DragonFly Doji !");
      return(1);
   }
   else
   {
      //Comment("Nooooooo DragonFly Doji " );
      return(0);
   }
   
}
//-----------------------------------------------------------------//
//--                      End DragonFly methods                  --// 
//-----------------------------------------------------------------//

//-----------------------------------------------------------------//
//--                          Engulfing methods                  --// 
//-----------------------------------------------------------------//
int FindBearishEngulfingMethod(string symbol, int timeFrame,  int candleNo, int engulfingMode)
{  
   //-- candleNo is the candle which is engulf the perviouse candle .
   //-- engulfing mode : 0=body engulf body || 1= body engulf body+shadow || 2= body engulf whole candle .
   double C=iClose(symbol,timeFrame,candleNo);  
   double O=iOpen(symbol,timeFrame,candleNo);
   double H=iHigh(symbol,timeFrame,candleNo);
   double L=iLow(symbol,timeFrame,candleNo);
   
   double C_1=iClose(symbol,timeFrame,candleNo+1);
   double O_1=iOpen(symbol,timeFrame,candleNo+1);
   double H_1=iHigh(symbol,timeFrame,candleNo+1);
   double L_1=iLow(symbol,timeFrame,candleNo+1);
     
   double spread_value = spread* Point;   
   
   switch(engulfingMode)
   {
      case 0:
         if(( C <O_1) && ((O+spread_value) > C_1)  && (C_1>O_1))     //(C - spread_value)    
            return(-1);      //------------------   Sell is occured !
         else
            return(0);
         
      case 1:
         if(( C<=O_1) && (O >= C_1)  && (C_1>O_1))
            return(-1);      //------------------   Sell is occured !
         else
            return(0);

      case 2:
         if(( (C+spread_value) <=L_1) && (O>=H_1)  && (C_1>O_1))
            return(-1);      //------------------   Sell is occured !
         else
            return(0);
      
      default:
      {
         //Alert("Your input parameter is not correct, please check again !");
         return(0);
      }   
   }   
   
}

//+------------------------------------------------------------------+
int FindBullishEngulfingMethod(string symbol, int timeFrame,  int candleNo, int engulfingMode)
{  
   //-- candleNo is the candle which is engulf the perviouse candle .
   //-- engulfing mode : 0=body engulf body || 1= body engulf body+shadow || 2= body engulf whole candle .
   double C=iClose(symbol,timeFrame,candleNo);  
   double O=iOpen(symbol,timeFrame,candleNo);
   double H=iHigh(symbol,timeFrame,candleNo);
   double L=iLow(symbol,timeFrame,candleNo);
   
   double C_1=iClose(symbol,timeFrame,candleNo+1);
   double O_1=iOpen(symbol,timeFrame,candleNo+1);
   double H_1=iHigh(symbol,timeFrame,candleNo+1);
   double L_1=iLow(symbol,timeFrame,candleNo+1);
    
   double spread_value = spread* Point;  
   //Alert("Spred: ",spread_value);
   //Comment("Spred: ",spread_value);
      
   switch(engulfingMode)
   {
      case 0:
         if(( C>O_1) && ( (O-spread_value) <C_1)  && (C_1<O_1))        
            return(1);      //------------------   Sell is occured !
         else
            return(0);
         
      case 1:
         if((C >= O_1) && ( O <= C_1)  && ( C_1 < O_1))
            return(1);      //------------------   Sell is occured !
         else
            return(0);

      case 2:
         if((C>=H_1) && (O<=L_1)  && (C_1<O_1))
            return(1);      //------------------   Sell is occured !
         else
            return(0);
      
      default:
      {
         //Alert("Your input parameter is not correct, please check again ENGULFING!");
         return(0);
      }   
   }   
}

//+------------------------------------------------------------------+
bool CalculateSLTP_Engulfing(string symbol, int orderType, int patternNo, int tpPercent)
{
   //-- SLTP_MorePipFromShadow is an input parameter that added in calculating SL & TP to the pattern SL. 
   //-- Pattern No is : "1 :BearishEngulffing" - "2 :BullishEngulfing" -  
   double cur_SL,cur_TP;
   double high_1 = iHigh(symbol, ExpertDefualt, 1);
   double high_2 = iHigh(symbol, ExpertDefualt, 2);
   double low_1 = iLow(symbol, ExpertDefualt, 1);
   double low_2 = iLow(symbol, ExpertDefualt, 2);
   
   switch (patternNo)
   {
      case 1:   //-- Bearish Engulfing.
      {
         if(high_1>high_2)
         {cur_SL=high_1+SLTP_MorePipFromShadow*Point;cur_TP = Bid - (high_1-Ask)*tpPercent/100;}
         else 
         {cur_SL=high_2+SLTP_MorePipFromShadow*Point;cur_TP = Bid - (high_2-Ask)*tpPercent/100;}; 
         
         SL = cur_SL;
         TP = cur_TP;
         return(true);                 
      }
      case 2:   //-- Bullish Engulfing.
      {
         if(low_1<low_2)
         {cur_SL=low_1-SLTP_MorePipFromShadow*Point;cur_TP = Ask + (Ask-low_1)*tpPercent/100;}
         else 
         {cur_SL=low_2-SLTP_MorePipFromShadow*Point;cur_TP = Ask + (Ask-low_2)*tpPercent/100;};               

         SL = cur_SL;
         TP = cur_TP;
         return(true);                 
      }
      default:
         return(false);
   }
}

//+------------------------------------------------------------------+
int BodyMoreThanX(string symbol, int candleNo, double minLengthOfBody, int timeframe)
{
   double C_Confirm=iClose(symbol,timeframe,candleNo);
   double O_Confirm=iOpen(symbol,timeframe,candleNo);
   double H_Confirm=iHigh(symbol,timeframe,candleNo);
   double L_Confirm=iLow(symbol,timeframe,candleNo);

   if((C_Confirm<O_Confirm)&&(O_Confirm-C_Confirm)>minLengthOfBody*Point)
      return(1);
   else if((C_Confirm>O_Confirm)&&(C_Confirm-O_Confirm)>minLengthOfBody*Point)
      return(-1);
   else
      return(0);      
      
}
//+------------------------------------------------------------------+  //-- Engulfing -- MIN 
// int BodyConfirm(string symbol, int candleNo, double MaxLenght, double MinLenght, int timeframe)
int BodyConfirm(string symbol, int candleNo, double MaxLenght, int timeframe)
{
   double C_Confirm=iClose(symbol,timeframe,candleNo);
   double O_Confirm=iOpen(symbol,timeframe,candleNo);
   double H_Confirm=iHigh(symbol,timeframe,candleNo);
   double L_Confirm=iLow(symbol,timeframe,candleNo);
   
   if((C_Confirm<O_Confirm)&&((O_Confirm-C_Confirm)<MaxLenght*Point) ) // && ((O_Confirm-C_Confirm)>MinLenght*Point))
      return(-1);
   if((C_Confirm>O_Confirm)&&((C_Confirm-O_Confirm)<MaxLenght*Point) ) // && ((C_Confirm-O_Confirm)>MinLenght*Point))
      return(1);      
   else
      return(0);
}
//-----------------------------------------------------------------//
//--                      End Engulfing methods                  --// 
//-----------------------------------------------------------------//


//------------------------------------------------------------------------------------//
//---------------------------- Function to write trade information to a CSV file -----//

//--------------------------------------------------------------------------------------------//


// Function to add trade information to the buffer
void AddTradeInfoToBuffer(datetime tradeTime, double price, string orderType, string pattern, double volume, double stopLoss, double takeProfit) {
   tradeDataBuffer += StringFormat(
      "%s,%s,%.5f,%s,%s,%.2f,%.5f,%.5f\r\n",
      TimeToString(tradeTime, "yyyyMMdd"),
      TimeToString(tradeTime, "hh:mm"),
      price,
      orderType,
      pattern,
      volume,
      stopLoss,
      takeProfit
   );
   
   //TimeToString(currentTime, "yyyyMMdd");
}

//-----------------------------------------
// Function to write trade information to a CSV file (writes the buffer)
bool WriteTradeInfoToCSV(string filename) {
   int fileHandle = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_CSV);

   if (fileHandle == INVALID_HANDLE) {
      Print("Error opening file: ", filename, " Error: ", GetLastError());
      return false;
   }

   long fileSize = FileSize(fileHandle);
   bool fileIsEmpty = (fileSize == 0);

   if (fileIsEmpty) {
      string header = "Date,Time,Price,Order Type,Pattern,Volume,Stop Loss,Take Profit\r\n";
      FileWriteString(fileHandle, header);
   }

   FileSeek(fileHandle, 0, SEEK_END); // Move to the end of the file

   bool writeResult = FileWriteString(fileHandle, tradeDataBuffer) > 0; // Write the buffer
   FileClose(fileHandle);

   tradeDataBuffer = ""; // *** CRUCIAL: Clear the buffer *AFTER* writing ***
   return writeResult;
}